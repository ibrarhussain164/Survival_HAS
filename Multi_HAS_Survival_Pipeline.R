# =============================================================================
# Mixed-Effects Hierarchical Aware Stacking (ME-HAS) — Survival Pipeline
# Script : Multi_HAS_Survival_Pipeline.R
# Version: 9
# Authors: Ibrar Hussain, Sohail Akhtar, Xinyi Cao, Shahzaib Rafiq,
#          Can Liu, Yiqing Gu, Zaixiang Tang
# Affil. : Soochow University / University of Tabuk
# =============================================================================
#
# Pipeline overview:
#   - Nine benchmark right-censored survival datasets
#   - Two missing-data mechanisms: MAR, MCAR
#   - Five missingness proportions: 10%, 20%, 30%, 40%, 50%
#   - S = 10 stratified random train/test splits per condition
#   - Methods: Complete / MeanSI / KNN / RFI / MissInd /
#              MIE-PMM-Avg / MIE-CART-Avg / MICE-SE / ME-HAS
#   - Metrics: Harrell C-index (primary), Uno's C-statistic (secondary)
#   - Bonferroni correction: 28 combos × 2 metrics × 5 rates = 280 per mechanism
#
# Design notes:
#   - Unified seed formula guarantees identical train/test partitions across
#     all methods within each experimental cell.
#   - Between-imputation random intercept (imp_id BLUP) is intentionally
#     excluded from test-set prediction; only the fixed slope and
#     classifier-specific BLUP are applied (see Section 2.6 of the paper).
#   - Split-level .rds checkpoints allow resumption after interruptions.
#   - MICE uses PMM with Nelson-Aalen cumulative hazard and event indicator
#     as passive auxiliary predictors (White & Royston, 2009).
#   - Missing Indicator: binary flag columns (_ind suffix) appended to
#     mean-imputed data; falls back to MeanSI if the model fails.
# =============================================================================

# ── 0. Packages ───────────────────────────────────────────────────────────────
needed <- c("mice","VIM","missForest","survival","gbm",
            "xgboost","coxme","pec","caret","dplyr","tidyr","ggplot2","purrr",
            "readr","flexsurv","scales")
missing_pkgs <- needed[!needed %in% installed.packages()[,"Package"]]
if (length(missing_pkgs) > 0) {
  message("Installing missing packages: ", paste(missing_pkgs, collapse=", "))
  install.packages(missing_pkgs, repos="https://cloud.r-project.org")
}
suppressPackageStartupMessages({
  library(mice); library(VIM); library(missForest)
  library(survival); library(gbm); library(xgboost)
  library(coxme); library(pec); library(flexsurv); library(caret)
  library(dplyr); library(tidyr); library(ggplot2)
  library(purrr); library(readr); library(scales)
})
set.seed(2024)

# ── 1. Settings ───────────────────────────────────────────────────────────────
N_SPLITS   <- 10
MISS_RATES <- seq(0.10, 0.50, by=0.10)
M_IMP      <- 5
M_IMP_SE   <- 5
MAXIT_MICE <- 20          # 20 Gibbs iterations for reliable convergence
CV_FOLDS   <- 5
TRAIN_FRAC <- 0.70
SURV_NAMES <- c("CoxPH", "GBM", "XGBoost", "WeibullAFT")
MISS_MECHS <- c("MAR", "MCAR")
METHOD_ORDER <- c("Complete","MeanSI","KNN","RFI","MissInd",
                  "MIE_PMM_AVG","MIE_CART_AVG","MICE_SE","ME_HAS")

# Bonferroni N computed dynamically after data loaded:
# n_combos × n_metrics × n_rates  (per mechanism, separate tables)
# Full run: 28 × 2 × 5 = 280 per mechanism

RUN_ID         <- "v9"
OUTPUT_DIR     <- paste0("surv_HAS_", RUN_ID)
CHECKPOINT_DIR <- file.path(OUTPUT_DIR, "checkpoints")
DATA_DIR       <- file.path(path.expand("~"), "survival_data")
DATA_DIR2      <- file.path(getwd(), "survival_data", "data")
dir.create(OUTPUT_DIR,     recursive=TRUE, showWarnings=FALSE)
dir.create(CHECKPOINT_DIR, recursive=TRUE, showWarnings=FALSE)
RESET_CHECKPOINTS <- FALSE
if (RESET_CHECKPOINTS && dir.exists(CHECKPOINT_DIR)) {
  old_ckpts <- list.files(CHECKPOINT_DIR, pattern="\\.rds$", full.names=TRUE)
  if (length(old_ckpts) > 0) {
    file.remove(old_ckpts)
    cat(sprintf(">>> RESET: deleted %d checkpoint(s)\n", length(old_ckpts)))
  }
}

# ── 2. Unified seed function ──────────────────────────────────────────────────
# One seed formula for ALL splits (complete + missing).
# Complete-case uses miss_rate=0 and mech="NONE" → offset=0.
# This guarantees all methods see the exact same train/test partition.
make_split_seed <- function(sp, ds_name, miss_rate=0, mech="NONE") {
  mech_offset <- switch(mech, MAR=9999L, MCAR=19999L, 0L)
  sp * 137L + as.integer(round(miss_rate * 1000)) + nchar(ds_name) + mech_offset
}

# ── 3. Dataset Loaders ────────────────────────────────────────────────────────
make_surv_ds <- function(X_num, time, event) {
  df <- data.frame(X_num)
  df$time  <- as.numeric(time)
  df$event <- as.integer(event)
  df <- df[complete.cases(df[, c("time","event")]) & df$time > 0, ]
  pred_cols <- setdiff(names(df), c("time","event"))
  df[, c(pred_cols[sapply(df[, pred_cols, drop=FALSE], is.numeric)], "time","event")]
}
load_csv <- function(filename) {
  candidates <- c(file.path(DATA_DIR, filename), file.path(DATA_DIR2, filename))
  fpath <- Find(file.exists, candidates)
  if (is.null(fpath)) stop(sprintf("CSV not found: %s", filename))
  d <- read.csv(fpath, stringsAsFactors=FALSE)
  pred_cols <- setdiff(names(d), c("time","event"))
  X <- d[, pred_cols, drop=FALSE]
  X <- X[, sapply(X, is.numeric), drop=FALSE]
  make_surv_ds(X, d$time, d$event)
}

ALL_DATASETS <- list(
  list(name="WHAS500",      loader=function() load_csv("WHAS500.csv")),
  list(name="GBSG2",        loader=function() load_csv("GBSG2.csv")),
  list(name="MGUS2",        loader=function() load_csv("MGUS2.csv")),
  list(name="WIHS",         loader=function() load_csv("WIHS.csv")),
  list(name="ACTG175",      loader=function() load_csv("ACTG175.csv")),
  list(name="PBC",          loader=function() load_csv("PBC.csv")),
  list(name="Colon",        loader=function() load_csv("Colon.csv")),
  list(name="FLChain",      loader=function() load_csv("FLChain.csv")),
  list(name="HeartFailure", loader=function() load_csv("HeartFailure.csv"))
)
DATASET_REGISTRY <- ALL_DATASETS

# ── 4. Missing Data Injection ─────────────────────────────────────────────────
induce_missing <- function(X, rate, mech="MAR",
                           time_vec=NULL, event_vec=NULL, seed=NULL) {
  if (rate == 0) return(X)
  if (!is.null(seed)) set.seed(seed)
  n <- nrow(X); p <- ncol(X); X_miss <- as.data.frame(X)
  if (toupper(mech) == "MAR") {
    beta_mar <- log(9)
    for (j in seq_len(p)) {
      others   <- setdiff(seq_len(p), j)
      aux_cols <- list()
      if (length(others) > 0) {
        Xoth <- scale(as.matrix(X[, others, drop=FALSE]))
        Xoth[!is.finite(Xoth)] <- 0
        aux_cols[[length(aux_cols)+1]] <- Xoth
      }
      if (!is.null(time_vec) && length(time_vec) == n) {
        lt <- scale(log(pmax(as.numeric(time_vec), 1e-6)))
        lt[!is.finite(lt)] <- 0
        aux_cols[[length(aux_cols)+1]] <- matrix(lt, ncol=1)
      }
      if (!is.null(event_vec) && length(event_vec) == n) {
        ev <- scale(as.numeric(event_vec))
        ev[!is.finite(ev)] <- 0
        aux_cols[[length(aux_cols)+1]] <- matrix(ev, ncol=1)
      }
      if (length(aux_cols) == 0) {
        X_miss[sample(n, max(1L, round(rate*n))), j] <- NA
        next
      }
      mar_score <- rowMeans(do.call(cbind, aux_cols))
      alpha_j <- tryCatch(
        uniroot(function(a) mean(plogis(a + beta_mar*mar_score)) - rate,
                interval=c(-50, 50), tol=1e-8)$root,
        error=function(e) qlogis(rate))
      X_miss[runif(n) < plogis(alpha_j + beta_mar*mar_score), j] <- NA
    }
  } else {
    for (j in seq_len(p))
      X_miss[sample(n, max(1L, round(rate*n)), replace=FALSE), j] <- NA
  }
  X_miss
}

# ── 5. Imputation ─────────────────────────────────────────────────────────────
impute_mean <- function(X_tr_miss, X_te_full=NULL) {
  fill_vals <- sapply(X_tr_miss, function(col) {
    fv <- mean(col, na.rm=TRUE); if (is.na(fv)) 0 else fv
  })
  fill_df <- function(df) {
    for (nm in names(df)) {
      idx <- is.na(df[[nm]]); if (!any(idx)) next; df[[nm]][idx] <- fill_vals[nm]
    }; df
  }
  list(train=fill_df(X_tr_miss),
       test=if (is.null(X_te_full)) NULL else X_te_full)
}

impute_knn <- function(X_tr_miss, X_te_full=NULL, k=5) {
  suppressMessages(suppressWarnings({
    list(train=VIM::kNN(X_tr_miss, k=k, imp_var=FALSE),
         test=if (is.null(X_te_full)) NULL else X_te_full)
  }))
}

impute_rfi <- function(X_tr_miss, X_te_full=NULL) {
  suppressMessages(suppressWarnings({
    list(train=missForest::missForest(X_tr_miss, verbose=FALSE)$ximp,
         test=if (is.null(X_te_full)) NULL else X_te_full)
  }))
}

# ── Missing Indicator imputation ──────────────────────────────────────────────
# For each column j that has any missing values in X_tr_miss:
#   1. Add binary column  <name>_ind  (1 = was missing, 0 = observed)
#      Applied to BOTH train and test (test missingness pattern = all zeros,
#      since test set is always complete in this pipeline).
#   2. Fill the missing values in column j with the training column mean.
# Result: p original columns + q indicator columns (q ≤ p).
# Training means and which columns got indicators are computed on train only
# and applied identically to test — no test information leaks.
impute_missind <- function(X_tr_miss, X_te_full=NULL) {
  col_means <- sapply(X_tr_miss, function(col) {
    mv <- mean(col, na.rm=TRUE); if (is.na(mv)) 0 else mv
  })
  missing_cols <- names(which(sapply(X_tr_miss, function(col) any(is.na(col)))))

  add_indicators <- function(df, ref_missing_cols, col_means_vec, is_train) {
    # Fill missing values with training means
    for (nm in names(df)) {
      idx <- is.na(df[[nm]]); if (!any(idx)) next
      df[[nm]][idx] <- col_means_vec[nm]
    }
    # Add indicator columns (1 = was missing in train; for test always 0)
    for (nm in ref_missing_cols) {
      ind_nm <- paste0(nm, "_ind")
      if (is_train) {
        # Already filled above; reconstruct indicator from original NAs
        # We saved which rows were NA before filling — use the original df
        # Actually we need to mark before filling; handled below
        df[[ind_nm]] <- 0L  # placeholder; overwritten for train
      } else {
        df[[ind_nm]] <- 0L  # test set is complete: all indicators = 0
      }
    }
    df
  }

  # Train: build indicator BEFORE filling
  train_out <- as.data.frame(X_tr_miss)
  ind_flags <- lapply(missing_cols, function(nm) as.integer(is.na(train_out[[nm]])))
  names(ind_flags) <- paste0(missing_cols, "_ind")
  # Fill means
  for (nm in names(train_out)) {
    idx <- is.na(train_out[[nm]]); if (!any(idx)) next
    train_out[[nm]][idx] <- col_means[nm]
  }
  # Append indicators
  for (nm in names(ind_flags)) train_out[[nm]] <- ind_flags[[nm]]

  # Test: complete → all indicators 0
  test_out <- if (is.null(X_te_full)) NULL else {
    te <- as.data.frame(X_te_full)
    for (nm in paste0(missing_cols, "_ind")) te[[nm]] <- 0L
    te
  }

  list(train=train_out, test=test_out,
       missing_cols=missing_cols, col_means=col_means)
}

mice_impute_surv <- function(X_miss, time_vec, event_vec,
                             m=M_IMP, seed=1, maxit=MAXIT_MICE) {
  n <- nrow(X_miss)
  na_H <- tryCatch({
    km_fit <- survival::survfit(survival::Surv(time_vec, event_vec) ~ 1, type="fh")
    sapply(seq_len(n), function(i) {
      ti <- time_vec[i]; if (is.na(ti) || ti <= 0) return(0)
      idx <- which(km_fit$time <= ti); if (length(idx) == 0) return(0)
      -log(max(km_fit$surv[max(idx)], 1e-9))
    })
  }, error=function(e) rep(0, n))
  df    <- data.frame(X_miss, .na_H=na_H, .time=time_vec, .event=event_vec)
  meths <- setNames(rep("pmm", ncol(df)), names(df))
  meths[c(".na_H",".time",".event")] <- ""
  for (nm in names(df)) if (!any(is.na(df[[nm]]))) meths[nm] <- ""
  pm <- mice::make.predictorMatrix(df)
  pm[c(".na_H",".time",".event"), ] <- 0L
  suppressMessages(suppressWarnings(
    mids <- mice::mice(df, m=m, method=meths, predictorMatrix=pm,
                       seed=seed, printFlag=FALSE, maxit=maxit)
  ))
  X_cols <- names(X_miss)
  lapply(seq_len(m), function(i) mice::complete(mids, i)[, X_cols, drop=FALSE])
}

mice_impute_std <- function(X_miss, m=M_IMP, seed=1, maxit=MAXIT_MICE) {
  meths <- setNames(rep("pmm", ncol(X_miss)), names(X_miss))
  for (nm in names(X_miss)) if (!any(is.na(X_miss[[nm]]))) meths[nm] <- ""
  suppressMessages(suppressWarnings(
    mids <- mice::mice(X_miss, m=m, method=meths, seed=seed,
                       printFlag=FALSE, maxit=maxit)
  ))
  lapply(seq_len(m), function(i) mice::complete(mids, i))
}

mice_impute_cart <- function(X_miss, time_vec, event_vec,
                             m=M_IMP, seed=1, maxit=MAXIT_MICE) {
  n <- nrow(X_miss)
  na_H <- tryCatch({
    km_fit <- survival::survfit(survival::Surv(time_vec, event_vec) ~ 1, type="fh")
    sapply(seq_len(n), function(i) {
      ti <- time_vec[i]; if (is.na(ti) || ti <= 0) return(0)
      idx <- which(km_fit$time <= ti); if (length(idx) == 0) return(0)
      -log(max(km_fit$surv[max(idx)], 1e-9))
    })
  }, error=function(e) rep(0, n))
  df    <- data.frame(X_miss, .na_H=na_H, .time=time_vec, .event=event_vec)
  meths <- setNames(rep("cart", ncol(df)), names(df))
  meths[c(".na_H",".time",".event")] <- ""
  for (nm in names(df)) if (!any(is.na(df[[nm]]))) meths[nm] <- ""
  pm <- mice::make.predictorMatrix(df)
  pm[c(".na_H",".time",".event"), ] <- 0L
  mids <- tryCatch(
    suppressMessages(suppressWarnings(
      mice::mice(df, m=m, method=meths, predictorMatrix=pm,
                 seed=seed, printFlag=FALSE, maxit=maxit))),
    error=function(e) NULL)
  if (is.null(mids)) {
    message("  [CART fallback → PMM]")
    return(mice_impute_surv(X_miss, time_vec, event_vec, m, seed, maxit))
  }
  X_cols <- names(X_miss)
  lapply(seq_len(m), function(i) mice::complete(mids, i)[, X_cols, drop=FALSE])
}

# ── 6. Survival Model Wrapper ─────────────────────────────────────────────────
fit_surv <- function(model_name, X_tr, time_tr, event_tr, X_te) {
  p_names <- names(X_tr)
  df_tr   <- data.frame(X_tr, time=time_tr, event=event_tr)
  df_te   <- as.data.frame(X_te)[, p_names, drop=FALSE]
  f_surv  <- as.formula(paste("Surv(time, event) ~", paste(p_names, collapse=" + ")))
  suppressWarnings(tryCatch({
    if (model_name == "CoxPH") {
      m <- survival::coxph(f_surv, data=df_tr, ties="efron",
                           control=coxph.control(iter.max=150, toler.inf=0.75))
      predict(m, newdata=df_te, type="lp")
    } else if (model_name == "GBM") {
      m <- gbm::gbm(f_surv, data=df_tr, distribution="coxph",
                    n.trees=200, shrinkage=0.05, interaction.depth=3,
                    verbose=FALSE, bag.fraction=0.632)
      gbm::predict.gbm(m, newdata=df_te, n.trees=200)
    } else if (model_name == "XGBoost") {
      X_mat  <- as.matrix(X_tr[, p_names, drop=FALSE])
      Xte_m  <- as.matrix(df_te[, p_names, drop=FALSE])
      label  <- ifelse(event_tr==1, time_tr, -time_tr)
      dtrain <- xgboost::xgb.DMatrix(data=X_mat, label=label)
      dtest  <- xgboost::xgb.DMatrix(data=Xte_m)
      params <- list(objective="survival:cox", eval_metric="cox-nloglik",
                     eta=0.05, max_depth=3, subsample=0.632,
                     colsample_bytree=0.8, lambda=1, alpha=0,
                     min_child_weight=5, nthread=1)
      m <- xgboost::xgb.train(params=params, data=dtrain, nrounds=200, verbose=0)
      as.numeric(predict(m, dtest))
    } else if (model_name == "WeibullAFT") {
      m <- survival::survreg(f_surv, data=df_tr, dist="weibull")
      -predict(m, newdata=df_te, type="lp")
    } else rep(NA_real_, nrow(df_te))
  }, error=function(e) rep(NA_real_, nrow(df_te))))
}

safe_fit_surv <- function(model_name, X_tr, time_tr, event_tr, X_te) {
  n_te   <- nrow(as.data.frame(X_te))
  n_tr_ok <- sum(!is.na(time_tr) & !is.na(event_tr))
  n_ev_ok <- sum(event_tr==1L, na.rm=TRUE)
  if (n_tr_ok < max(ncol(as.data.frame(X_tr))+2L, 5L) || n_ev_ok < 2L)
    return(rep(NA_real_, n_te))
  out <- tryCatch(fit_surv(model_name, X_tr, time_tr, event_tr, X_te),
                  error=function(e) rep(NA_real_, n_te))
  out <- as.numeric(out)
  if (length(out) != n_te) rep(NA_real_, n_te) else out
}

# ── 7. Survival Metrics ───────────────────────────────────────────────────────
compute_surv_metrics <- function(risk_scores, time_te, event_te,
                                 time_tr=NULL, event_tr=NULL) {
  risk_scores <- as.numeric(risk_scores)
  risk_scores[is.infinite(risk_scores)] <- NA_real_

  cindex <- tryCatch(
    survival::concordance(
      survival::Surv(time_te, event_te) ~ I(-risk_scores))$concordance,
    error=function(e) NA_real_)

  uno_c <- tryCatch({
    ev_times <- time_te[event_te == 1L & !is.na(time_te)]
    tau <- if (length(ev_times) >= 4L)
      quantile(ev_times, 0.75, na.rm=TRUE)
    else
      quantile(time_te[!is.na(time_te)], 0.75, na.rm=TRUE)
    if (!is.finite(tau) || tau <= 0) stop("invalid tau")
    df_te <- data.frame(time=time_te, event=as.integer(event_te))
    ci <- pec::cindex(
      object     = list(Model=risk_scores),
      formula    = Surv(time, event) ~ 1,
      data       = df_te,
      eval.times = tau,
      verbose    = FALSE)
    as.numeric(ci$AppCindex[["Model"]])
  }, error=function(e) {
    tryCatch({
      ev_times <- time_te[event_te == 1L & !is.na(time_te)]
      tau <- if (length(ev_times) >= 4L) quantile(ev_times, 0.75, na.rm=TRUE)
      else quantile(time_te[!is.na(time_te)], 0.75, na.rm=TRUE)
      survival::concordance(
        survival::Surv(time_te, event_te) ~ I(-risk_scores),
        timewt="n/G2", tau=tau)$concordance
    }, error=function(e2) NA_real_)
  })

  c(Cindex=as.numeric(cindex), Uno_C=as.numeric(uno_c))
}

# ── 8. MIE_PMM_AVG / MIE_CART_AVG ────────────────────────────────────────────
run_mie_avg_surv <- function(surv_name, imp_tr, time_tr, event_tr, imp_te) {
  m   <- length(imp_tr)
  mat <- matrix(NA_real_, nrow(imp_te[[1]]), m)
  for (k in seq_len(m))
    mat[, k] <- safe_fit_surv(surv_name, imp_tr[[k]], time_tr, event_tr, imp_te[[k]])
  list(risk=rowMeans(mat, na.rm=TRUE), risk_mat=mat)
}

# ── 9. OOF Risk Scores ────────────────────────────────────────────────────────
get_oof_surv <- function(surv_name, imp_tr, time_tr, event_tr, cv_folds=CV_FOLDS) {
  n <- length(time_tr); m <- length(imp_tr)
  oof   <- matrix(NA_real_, n, m)
  k_use <- min(cv_folds, sum(event_tr), n-1L)
  if (k_use < 2) return(oof)
  folds <- caret::createFolds(event_tr, k=k_use, list=TRUE, returnTrain=FALSE)
  for (k in seq_len(m))
    for (fi in folds) {
      tri     <- setdiff(seq_len(n), fi)
      oof[fi, k] <- tryCatch(
        safe_fit_surv(surv_name,
                      imp_tr[[k]][tri, , drop=FALSE],
                      time_tr[tri], event_tr[tri],
                      imp_tr[[k]][fi, , drop=FALSE]),
        error=function(e) rep(NA_real_, length(fi)))
    }
  oof
}

# ── 10. ME-HAS Meta-Learner ───────────────────────────────────────────────────
# The between-imputation random intercept (imp_id BLUP) is intentionally
# excluded from test-set predictions.
#
# Reason: test_list[[clf]][, k] = model trained on imp_tr[[k]] applied to
# X_te_full (complete, identical across k). The label "imp_id=k" on the test
# side refers to which training imputation produced the model — it does NOT
# correspond to a property of the test observations themselves. Applying the
# training-estimated BLUP v_m to test rows whose imp label is only an
# arbitrary index would introduce a spurious, direction-unknown adjustment.
#
# Practical impact: since var_intercept_imp << var_slope_clf (grand-mean
# ratio 33:1), the re_imp values are near-zero and the fix has negligible
# effect on C-index. Nevertheless, the corrected formula is theoretically
# clean: only the fixed slope beta and the classifier BLUP u_c are used,
# and averaging over c and m in Eq.(mehas_final) produces the final score.

run_has_surv <- function(oof_list, test_list, time_tr, event_tr) {
  clfs  <- names(oof_list)
  m     <- ncol(oof_list[[1]])
  n_tr  <- length(time_tr)
  n_te  <- nrow(test_list[[1]])

  # ── Build long-format training data ──
  long_tr <- do.call(rbind, lapply(clfs, function(cl)
    do.call(rbind, lapply(seq_len(m), function(k)
      data.frame(lp     = oof_list[[cl]][, k],
                 time   = time_tr,
                 event  = event_tr,
                 obs_id = factor(seq_len(n_tr)),
                 imp_id = factor(k),
                 clf_id = factor(cl, levels=clfs))))))

  # ── Per-classifier normalisation (training moments) ──
  clf_norm <- setNames(lapply(clfs, function(cl) {
    vals <- long_tr$lp[long_tr$clf_id == cl]
    mu   <- mean(vals, na.rm=TRUE)
    s    <- sd(vals,   na.rm=TRUE)
    if (is.na(s) || s < 1e-8) s <- 1
    list(mu=mu, s=s)
  }), clfs)
  for (cl in clfs) {
    idx <- long_tr$clf_id == cl
    long_tr$lp[idx] <- (long_tr$lp[idx] - clf_norm[[cl]]$mu) / clf_norm[[cl]]$s
  }
  long_tr$lp[!is.finite(long_tr$lp)] <- 0

  # ── Fit coxme with fallback hierarchy ──
  ok <- FALSE; mod <- NULL
  for (frm in list(
    Surv(time,event) ~ lp + (lp|clf_id) + (1|imp_id),
    Surv(time,event) ~ lp + (1 |clf_id) + (1|imp_id),
    Surv(time,event) ~ lp + (1 |clf_id)
  )) {
    if (ok) break
    suppressMessages(suppressWarnings(tryCatch({
      mod <- coxme::coxme(frm, data=long_tr); ok <- TRUE
    }, error=function(e) NULL)))
  }

  # ── Build long-format test data ──
  long_te <- do.call(rbind, lapply(clfs, function(cl)
    do.call(rbind, lapply(seq_len(m), function(k)
      data.frame(
        lp      = (test_list[[cl]][, k] - clf_norm[[cl]]$mu) / clf_norm[[cl]]$s,
        obs_id  = factor(seq_len(n_te)),
        imp_id  = factor(k),
        clf_id  = factor(cl, levels=clfs),
        obs_int = seq_len(n_te))))))
  long_te$lp[!is.finite(long_te$lp)] <- 0

  # ── Fallback: no coxme, use row-mean ──
  if (!ok) {
    agg <- aggregate(lp ~ obs_int, long_te, mean)
    return(list(pred    = agg[order(agg$obs_int), "lp"],
                varcomp = list(var_slope_clf=NA_real_, var_intercept_imp=NA_real_)))
  }

  # ── Extract fixed effect ──
  beta_lp <- tryCatch({
    fe <- coxme::fixef(mod)
    v  <- if ("lp" %in% names(fe)) fe["lp"] else fe[1]
    if (is.finite(v)) v else 1
  }, error=function(e) 1)

  # ── Extract random effects ──
  re <- tryCatch(coxme::ranef(mod), error=function(e) list())
  get_re <- function(nm, levs) {
    if (!nm %in% names(re)) return(setNames(rep(0, length(levs)), levs))
    rv <- re[[nm]]
    if (is.data.frame(rv))     rv <- rv[, 1, drop=TRUE]
    else if (is.matrix(rv))    rv <- rv[, 1]
    else                       rv <- unlist(rv)
    vals <- setNames(rep(0, length(levs)), levs)
    common <- intersect(names(rv), levs)
    vals[common] <- rv[common]
    vals
  }
  re_clf <- get_re("clf_id", clfs)
  # re_imp (v_m) is intentionally excluded from test-set prediction;
  # only the fixed slope (beta) and classifier BLUP (u_c) are applied.

  # ── Test-set prediction: beta*lp + u_c only ──
  # v_m is dropped for test observations: averaging over M imputations
  # is equivalent to marginalising over the between-imputation random effect.
  long_te$pred <- beta_lp * long_te$lp +
    re_clf[as.character(long_te$clf_id)]
  long_te$pred[!is.finite(long_te$pred)] <- 0

  agg      <- aggregate(pred ~ obs_int, long_te, mean)
  pred_vec <- agg[order(agg$obs_int), "pred"]

  varcomp <- extract_coxme_varcomp(mod)
  list(pred=pred_vec, varcomp=varcomp)
}

extract_coxme_varcomp <- function(mod) {
  empty <- list(var_slope_clf=NA_real_, var_intercept_imp=NA_real_)
  if (is.null(mod)) return(empty)
  vc <- tryCatch(coxme::VarCorr(mod), error=function(e) list())
  if (length(vc) == 0) return(empty)
  get_var <- function(nm) {
    if (!nm %in% names(vc)) return(NA_real_)
    m <- vc[[nm]]
    if (is.matrix(m) && nrow(m)>=1 && ncol(m)>=1) return(as.numeric(m[1,1]))
    if (is.numeric(m) && length(m)>=1)             return(as.numeric(m[1]))
    NA_real_
  }
  list(var_slope_clf    = get_var("clf_id"),
       var_intercept_imp = get_var("imp_id"))
}

# ── 11. MICE_SE ───────────────────────────────────────────────────────────────
mice_se_l0_surv <- function(imp_tr, time_tr, event_tr, imp_te,
                            m_se=M_IMP_SE, oof_folds=CV_FOLDS) {
  m_use <- min(m_se, length(imp_tr))
  n_tr  <- length(time_tr); n_te <- nrow(imp_te[[1]])
  k_use <- min(oof_folds, n_tr)
  folds <- caret::createFolds(event_tr, k=k_use, list=TRUE, returnTrain=FALSE)

  L0_tr_sum <- matrix(0,  n_tr, length(SURV_NAMES), dimnames=list(NULL, SURV_NAMES))
  L0_te_sum <- matrix(0,  n_te, length(SURV_NAMES), dimnames=list(NULL, SURV_NAMES))
  L0_tr_cnt <- matrix(0L, n_tr, length(SURV_NAMES), dimnames=list(NULL, SURV_NAMES))
  L0_te_cnt <- matrix(0L, n_te, length(SURV_NAMES), dimnames=list(NULL, SURV_NAMES))

  for (sn in SURV_NAMES)
    for (k in seq_len(m_use)) {
      oof_k <- rep(NA_real_, n_tr)
      for (fi in folds) {
        tri     <- setdiff(seq_len(n_tr), fi)
        oof_k[fi] <- tryCatch(
          safe_fit_surv(sn,
                        imp_tr[[k]][tri, , drop=FALSE],
                        time_tr[tri], event_tr[tri],
                        imp_tr[[k]][fi, , drop=FALSE]),
          error=function(e) rep(NA_real_, length(fi)))
      }
      v_tr <- !is.na(oof_k) & is.finite(oof_k)
      L0_tr_sum[v_tr, sn] <- L0_tr_sum[v_tr, sn] + oof_k[v_tr]
      L0_tr_cnt[v_tr, sn] <- L0_tr_cnt[v_tr, sn] + 1L

      te_k <- tryCatch(
        safe_fit_surv(sn, imp_tr[[k]], time_tr, event_tr, imp_te[[k]]),
        error=function(e) rep(NA_real_, n_te))
      v_te <- !is.na(te_k) & is.finite(te_k)
      L0_te_sum[v_te, sn] <- L0_te_sum[v_te, sn] + te_k[v_te]
      L0_te_cnt[v_te, sn] <- L0_te_cnt[v_te, sn] + 1L
    }

  L0_tr <- ifelse(L0_tr_cnt > 0L, L0_tr_sum / L0_tr_cnt, 0)
  L0_te <- ifelse(L0_te_cnt > 0L, L0_te_sum / L0_te_cnt, 0)
  list(L0_tr=L0_tr, L0_te=L0_te)
}

run_mice_se_surv <- function(L0_tr, L0_te, time_tr, event_tr, meta_surv) {
  df_tr <- as.data.frame(L0_tr); df_te <- as.data.frame(L0_te)
  df_tr$time <- time_tr; df_tr$event <- event_tr
  lp_terms <- paste(names(as.data.frame(L0_tr)), collapse=" + ")
  f_meta   <- as.formula(paste("Surv(time, event) ~", lp_terms))
  tryCatch({
    if (meta_surv == "CoxPH") {
      m <- survival::coxph(f_meta, data=df_tr, ties="efron",
                           control=coxph.control(iter.max=150, toler.inf=0.75))
      predict(m, newdata=df_te, type="lp")
    } else {
      safe_fit_surv(meta_surv, as.data.frame(L0_tr), time_tr, event_tr,
                    as.data.frame(L0_te))
    }
  }, error=function(e) rowMeans(L0_te, na.rm=TRUE))
}

# ── 12. Single Run ────────────────────────────────────────────────────────────
run_one_surv <- function(miss_mech, miss_rate, split_seed,
                         X_tr_full, time_tr, event_tr,
                         X_te_full, time_te, event_te) {
  rows <- list()

  rec <- function(method, surv_name, pred) {
    mt <- compute_surv_metrics(pred, time_te, event_te)
    data.frame(Method=method, Classifier=surv_name, t(mt), stringsAsFactors=FALSE)
  }

  X_tr_miss <- tryCatch(
    induce_missing(X_tr_full, miss_rate, miss_mech,
                   time_vec=time_tr, event_vec=event_tr),
    error=function(e) X_tr_full)

  # ── Single-imputation baselines ──
  fb <- list()
  si_mean <- impute_mean(X_tr_miss, X_te_full)
  fb$knn  <- FALSE
  si_knn  <- tryCatch(
    impute_knn(X_tr_miss, X_te_full),
    error=function(e) {
      fb$knn <<- TRUE
      message(sprintf("  [FALLBACK] KNN → MeanSI: %s", conditionMessage(e)))
      impute_mean(X_tr_miss, X_te_full)
    })
  fb$rfi  <- FALSE
  si_rfi  <- tryCatch(
    impute_rfi(X_tr_miss, X_te_full),
    error=function(e) {
      fb$rfi <<- TRUE
      message(sprintf("  [FALLBACK] RFI → MeanSI: %s", conditionMessage(e)))
      impute_mean(X_tr_miss, X_te_full)
    })

  # ── Missing Indicator ──
  fb$missind <- FALSE
  si_missind <- tryCatch(
    impute_missind(X_tr_miss, X_te_full),
    error=function(e) {
      fb$missind <<- TRUE
      message(sprintf("  [FALLBACK] MissInd → MeanSI: %s", conditionMessage(e)))
      impute_mean(X_tr_miss, X_te_full)
    })

  for (sn in SURV_NAMES) {
    rows[[length(rows)+1]] <- rec("MeanSI", sn,
                                  safe_fit_surv(sn, si_mean$train,   time_tr, event_tr, si_mean$test))
    rows[[length(rows)+1]] <- rec("KNN", sn,
                                  safe_fit_surv(sn, si_knn$train,    time_tr, event_tr, si_knn$test))
    rows[[length(rows)+1]] <- rec("RFI", sn,
                                  safe_fit_surv(sn, si_rfi$train,    time_tr, event_tr, si_rfi$test))
    # MissInd: model trained on (original cols + indicator cols)
    # Fallback: if model fails with expanded feature set, safe_fit_surv
    # returns NA → handled gracefully by compute_surv_metrics
    rows[[length(rows)+1]] <- rec("MissInd", sn,
                                  safe_fit_surv(sn, si_missind$train, time_tr, event_tr, si_missind$test))
  }

  # ── MICE PMM + ME-HAS OOF ──
  fb$mice_na <- FALSE
  imp_tr <- tryCatch(
    mice_impute_surv(X_tr_miss, time_tr, event_tr, M_IMP, split_seed, MAXIT_MICE),
    error=function(e) {
      fb$mice_na <<- TRUE
      message(sprintf("  [FALLBACK] MICE+NA → std MICE: %s", conditionMessage(e)))
      mice_impute_std(X_tr_miss, M_IMP, split_seed, MAXIT_MICE)
    })
  imp_te <- lapply(seq_len(M_IMP), function(i) X_te_full)

  oof_list <- list(); test_list <- list()
  for (sn in SURV_NAMES) {
    oof_list[[sn]] <- tryCatch(
      get_oof_surv(sn, imp_tr, time_tr, event_tr),
      error=function(e) matrix(0, length(time_tr), M_IMP))
    res <- run_mie_avg_surv(sn, imp_tr, time_tr, event_tr, imp_te)
    rows[[length(rows)+1]] <- rec("MIE_PMM_AVG", sn, res$risk)
    test_list[[sn]] <- res$risk_mat
  }

  # ── MIE CART AVG ──
  fb$mice_cart  <- FALSE
  imp_tr_cart   <- tryCatch(
    mice_impute_cart(X_tr_miss, time_tr, event_tr, M_IMP, split_seed+2L, MAXIT_MICE),
    error=function(e) {
      fb$mice_cart <<- TRUE
      message(sprintf("  [FALLBACK] MIE_CART_AVG → reuse PMM imp: %s", conditionMessage(e)))
      imp_tr
    })
  imp_te_cart <- lapply(seq_len(M_IMP), function(i) X_te_full)
  for (sn in SURV_NAMES) {
    res_cart <- run_mie_avg_surv(sn, imp_tr_cart, time_tr, event_tr, imp_te_cart)
    rows[[length(rows)+1]] <- rec("MIE_CART_AVG", sn, res_cart$risk)
  }

  # ── MICE_SE ──
  fb$mice_se_imp <- FALSE
  imp_tr_se <- tryCatch(
    mice_impute_surv(X_tr_miss, time_tr, event_tr, M_IMP_SE, split_seed+1L, MAXIT_MICE),
    error=function(e) {
      fb$mice_se_imp <<- TRUE
      message(sprintf("  [FALLBACK] MICE_SE imp → reuse imp_tr: %s", conditionMessage(e)))
      imp_tr
    })
  imp_te_se  <- lapply(seq_len(M_IMP_SE), function(i) X_te_full)
  fb$mice_se_l0 <- FALSE
  l0 <- tryCatch(
    mice_se_l0_surv(imp_tr_se, time_tr, event_tr, imp_te_se),
    error=function(e) {
      fb$mice_se_l0 <<- TRUE
      message(sprintf("  [FALLBACK] MICE_SE L0 → rowMeans: %s", conditionMessage(e)))
      NULL
    })
  for (meta_sn in SURV_NAMES) {
    risk_se <- if (!is.null(l0)) {
      tryCatch(run_mice_se_surv(l0$L0_tr, l0$L0_te, time_tr, event_tr, meta_sn),
               error=function(e) rowMeans(l0$L0_te, na.rm=TRUE))
    } else {
      rowMeans(do.call(cbind, lapply(seq_len(M_IMP_SE), function(k)
        safe_fit_surv(meta_sn, imp_tr_se[[k]], time_tr, event_tr, X_te_full))),
        na.rm=TRUE)
    }
    rows[[length(rows)+1]] <- rec("MICE_SE", meta_sn, risk_se)
  }

  # ── ME-HAS ──
  fb$me_has  <- FALSE
  mehas_t0   <- proc.time()
  has_out <- tryCatch(
    run_has_surv(oof_list, test_list, time_tr, event_tr),
    error=function(e) {
      fb$me_has <<- TRUE
      message(sprintf("  [FALLBACK] ME_HAS → rowMeans: %s", conditionMessage(e)))
      list(pred    = rowMeans(do.call(cbind, lapply(test_list, rowMeans, na.rm=TRUE))),
           varcomp = list(var_slope_clf=NA_real_, var_intercept_imp=NA_real_))
    })
  mehas_elapsed <- (proc.time() - mehas_t0)[["elapsed"]]
  rows[[length(rows)+1]] <- rec("ME_HAS", "All", has_out$pred)
  me_has_vc <- has_out$varcomp

  # ── Variance decomposition rows ──
  var_rows <- do.call(rbind, lapply(SURV_NAMES, function(sn) {
    mat <- oof_list[[sn]]
    if (is.null(mat) || ncol(mat) < 2) return(NULL)
    subj_sd <- apply(mat, 1, sd, na.rm=TRUE)
    data.frame(MissMech=miss_mech, MissRate=miss_rate, Classifier=sn,
               mean_lp_sd=mean(subj_sd, na.rm=TRUE),
               var_slope_clf=NA_real_, var_intercept_imp=NA_real_,
               stringsAsFactors=FALSE)
  }))
  var_rows <- rbind(var_rows, data.frame(
    MissMech=miss_mech, MissRate=miss_rate, Classifier="ME_HAS",
    mean_lp_sd=NA_real_,
    var_slope_clf    = me_has_vc$var_slope_clf,
    var_intercept_imp = me_has_vc$var_intercept_imp,
    stringsAsFactors=FALSE))

  fb_row <- data.frame(
    MissMech       = miss_mech,
    MissRate       = miss_rate,
    fb_knn         = fb$knn,
    fb_rfi         = fb$rfi,
    fb_missind     = fb$missind,
    fb_mice_na     = fb$mice_na,
    fb_mice_cart   = fb$mice_cart,
    fb_mice_se_imp = fb$mice_se_imp,
    fb_mice_se_l0  = fb$mice_se_l0,
    fb_me_has      = fb$me_has,
    stringsAsFactors=FALSE)

  list(rows=do.call(rbind, rows), var_rows=var_rows,
       fb_row=fb_row, mehas_elapsed=mehas_elapsed)
}

# ── 13. Split-level checkpoint helpers ────────────────────────────────────────
# Granular checkpoints: one .rds per (dataset, mech, rate, split).
# If a run crashes mid-dataset, only the interrupted split is re-run.
# A dataset-level summary .rds is also written after each full dataset
# for fast validation / re-loading.

split_ckpt_file <- function(ds_name, mech, miss_rate, sp) {
  rate_tag <- sprintf("%02d", as.integer(round(miss_rate * 100)))
  fname    <- sprintf("%s_%s_%s_%02d.rds", ds_name, mech, rate_tag, sp)
  file.path(CHECKPOINT_DIR, fname)
}

complete_ckpt_file <- function(ds_name, sp) {
  fname <- sprintf("%s_complete_%02d.rds", ds_name, sp)
  file.path(CHECKPOINT_DIR, fname)
}

ds_ckpt_file <- function(ds_name) {
  file.path(CHECKPOINT_DIR, paste0(ds_name, ".rds"))
}

save_split_ckpt <- function(path, rows, var_rows, fb_row,
                            miss_log_row, mehas_elapsed, elapsed_sec) {
  saveRDS(list(rows=rows, var_rows=var_rows, fb_row=fb_row,
               miss_log_row=miss_log_row,
               mehas_elapsed=mehas_elapsed,
               elapsed_sec=elapsed_sec), path)
}

load_split_ckpt <- function(path) {
  if (!file.exists(path)) return(NULL)
  tryCatch({
    ck <- readRDS(path)
    # Pad any missing columns introduced by new methods (e.g. fb_missind)
    if (!is.null(ck$rows))
      ck$rows <- pad_df(ck$rows, RESULT_COLS)
    if (!is.null(ck$fb_row))
      ck$fb_row <- pad_df(ck$fb_row, FB_COLS)
    ck
  }, error=function(e) {
    message(sprintf("  [WARN] corrupt split ckpt, will rerun: %s", basename(path)))
    NULL
  })
}

# ── 14. Main Experiment Loop ──────────────────────────────────────────────────
cat(sprintf("\n=== ME-HAS Survival Benchmark (%d datasets) ===\n", length(DATASET_REGISTRY)))
cat("Models: CoxPH / GBM / XGBoost / WeibullAFT\n")
cat("Baselines: Complete/MeanSI/KNN/RFI/MissInd/MIE_PMM_AVG/MIE_CART_AVG/MICE_SE\n")
cat("Metrics: Harrell C-index (primary)  |  Uno's C-statistic (secondary, IPCW)\n")
cat(sprintf("Splits: %d | M_IMP: %d | MAXIT_MICE: %d\n", N_SPLITS, M_IMP, MAXIT_MICE))
cat("Checkpoints: split-level (dataset × mech × rate × split)\n\n")

all_results         <- list()
all_var_rows        <- list()
all_fb_rows         <- list()
all_miss_log        <- list()
all_time_rows       <- list()
all_mehas_time_rows <- list()
pipeline_t0         <- proc.time()

RESULT_COLS <- c("Method","Classifier","Cindex","Uno_C",
                 "Dataset","MissMech","MissRate","Split","ActualMiss")
FB_COLS     <- c("MissMech","MissRate",
                 "fb_knn","fb_rfi","fb_missind","fb_mice_na","fb_mice_cart",
                 "fb_mice_se_imp","fb_mice_se_l0","fb_me_has")

pad_df <- function(df, required_cols) {
  for (col in setdiff(required_cols, names(df))) df[[col]] <- NA_real_
  df
}
harmonise_ckpt <- function(ckpt) {
  ckpt$results <- lapply(ckpt$results, pad_df, required_cols=RESULT_COLS)
  ckpt$fb_rows <- lapply(ckpt$fb_rows, pad_df, required_cols=FB_COLS)
  ckpt
}

for (ds in DATASET_REGISTRY) {

  # ── Check dataset-level summary ckpt (fast path) ──────────────────────────
  ds_ckpt <- ds_ckpt_file(ds$name)
  if (file.exists(ds_ckpt)) {
    ckpt_raw        <- tryCatch(readRDS(ds_ckpt), error=function(e) NULL)
    if (!is.null(ckpt_raw)) {
      mechs_in_ckpt   <- unique(unlist(lapply(ckpt_raw$results, function(df)
        if (!is.null(df$MissMech)) unique(df$MissMech) else character(0))))
      missing_mechs   <- setdiff(MISS_MECHS, mechs_in_ckpt)
      has_brier       <- any(unlist(lapply(ckpt_raw$results,
                                           function(df) "Brier" %in% names(df))))
      has_uno_c       <- any(unlist(lapply(ckpt_raw$results,
                                           function(df) "Uno_C" %in% names(df))))
      methods_in_ckpt <- unique(unlist(lapply(ckpt_raw$results, function(df)
        if (!is.null(df$Method)) unique(df$Method) else character(0))))
      stale_methods   <- length(setdiff(METHOD_ORDER, methods_in_ckpt)) > 0 ||
        length(setdiff(methods_in_ckpt, c(METHOD_ORDER,"Complete"))) > 0
      clfs_in_ckpt    <- unique(unlist(lapply(ckpt_raw$results, function(df)
        if (!is.null(df$Classifier)) unique(df$Classifier) else character(0))))
      stale_clfs      <- length(setdiff(SURV_NAMES, clfs_in_ckpt)) > 0 ||
        length(setdiff(clfs_in_ckpt, c(SURV_NAMES,"All"))) > 0
      ds_stale <- length(missing_mechs)>0 || has_brier || !has_uno_c ||
        stale_methods || stale_clfs
      if (!ds_stale) {
        ckpt         <- harmonise_ckpt(ckpt_raw)
        all_results  <- c(all_results,  ckpt$results)
        all_var_rows <- c(all_var_rows, ckpt$var_rows)
        all_fb_rows  <- c(all_fb_rows,  ckpt$fb_rows)
        all_miss_log <- c(all_miss_log, ckpt$miss_log)
        cat(sprintf(">>> %s  [DS-CKPT loaded — skip]\n", ds$name))
        next
      }
      # Stale ds-ckpt: delete it but keep any valid split ckpts below
      reason <- paste(c(
        if (length(missing_mechs)>0) paste("missing mech:", paste(missing_mechs,collapse=",")),
        if (has_brier)     "Brier checkpoint",
        if (!has_uno_c)    "missing Uno_C",
        if (stale_methods) "method list changed",
        if (stale_clfs)    sprintf("clf list changed (%s)", paste(clfs_in_ckpt,collapse=","))
      ), collapse="; ")
      cat(sprintf(">>> %s  [DS-CKPT stale (%s) — using split ckpts]\n", ds$name, reason))
      file.remove(ds_ckpt)
    }
  }

  # ── Load dataset ──────────────────────────────────────────────────────────
  cat(sprintf(">>> Processing: %s\n", ds$name))
  DATASET <- tryCatch(ds$loader(), error=function(e) {
    message(sprintf("  [SKIP] %s: %s", ds$name, conditionMessage(e))); NULL })
  if (is.null(DATASET) || nrow(DATASET) < 50) {
    cat(sprintf("  [SKIP] %s — n=%s\n", ds$name,
                if (is.null(DATASET)) "NA" else nrow(DATASET)))
    next
  }
  PRED_NAMES <- setdiff(names(DATASET), c("time","event"))
  non_num    <- PRED_NAMES[!sapply(DATASET[, PRED_NAMES, drop=FALSE], is.numeric)]
  if (length(non_num) > 0) {
    cat(sprintf("  [SKIP] %s — non-numeric: %s\n",
                ds$name, paste(non_num, collapse=", ")))
    next
  }
  n_events <- sum(DATASET$event, na.rm=TRUE)
  EPV      <- round(n_events / max(length(PRED_NAMES), 1), 1)
  cat(sprintf("  n=%d  p=%d  events=%d (%.0f%%)  EPV=%.1f%s\n",
              nrow(DATASET), length(PRED_NAMES), n_events,
              100*n_events/nrow(DATASET), EPV,
              if (EPV < 10) " [!EPV<10]" else ""))

  ds_results <- list(); ds_var_rows <- list()
  ds_fb_rows <- list(); ds_miss_log <- list()
  ds_t0      <- proc.time()

  # ── Complete-case baseline (split-level ckpts) ────────────────────────────
  n_complete_loaded <- 0L
  for (sp in seq_len(N_SPLITS)) {
    cc_ckpt <- complete_ckpt_file(ds$name, sp)
    cc_data <- load_split_ckpt(cc_ckpt)
    if (!is.null(cc_data)) {
      # Already done — load from split ckpt
      ds_results <- c(ds_results, list(cc_data$rows))
      n_complete_loaded <- n_complete_loaded + 1L
      next
    }
    # Run complete-case for this split
    base_seed <- make_split_seed(sp, ds$name, miss_rate=0, mech="NONE")
    set.seed(base_seed)
    tr_idx0 <- caret::createDataPartition(DATASET$event, p=TRAIN_FRAC, list=FALSE)[, 1]
    te_idx0 <- setdiff(seq_len(nrow(DATASET)), tr_idx0)
    X_tr0   <- DATASET[tr_idx0, PRED_NAMES, drop=FALSE]
    ttr0    <- DATASET[tr_idx0, "time"];  etr0 <- DATASET[tr_idx0, "event"]
    X_te0   <- DATASET[te_idx0, PRED_NAMES, drop=FALSE]
    tte0    <- DATASET[te_idx0, "time"];  ete0 <- DATASET[te_idx0, "event"]
    sp_t0_cc <- proc.time()
    b_rows <- list()
    for (mech0 in MISS_MECHS) {
      b <- do.call(rbind, lapply(SURV_NAMES, function(sn) {
        risk <- safe_fit_surv(sn, X_tr0, ttr0, etr0, X_te0)
        mt   <- compute_surv_metrics(risk, tte0, ete0)
        data.frame(Method="Complete", Classifier=sn, t(mt), stringsAsFactors=FALSE)
      }))
      b$Dataset    <- ds$name; b$MissMech  <- mech0
      b$MissRate   <- 0;       b$Split     <- sp
      b$ActualMiss <- 0
      b_rows[[length(b_rows)+1]] <- b
    }
    cc_rows       <- do.call(rbind, b_rows)
    sp_elapsed_cc <- (proc.time() - sp_t0_cc)[["elapsed"]]
    # Save split ckpt for complete case
    save_split_ckpt(cc_ckpt, rows=cc_rows,
                    var_rows=NULL, fb_row=NULL,
                    miss_log_row=NULL,
                    mehas_elapsed=0, elapsed_sec=sp_elapsed_cc)
    ds_results <- c(ds_results, list(cc_rows))
  }
  if (n_complete_loaded > 0)
    cat(sprintf("  [complete-case: %d/%d splits from ckpt]\n",
                n_complete_loaded, N_SPLITS))

  # ── Missing-data experiments (split-level ckpts) ──────────────────────────
  for (mech in MISS_MECHS) {
    cat(sprintf("  [%s]", mech))
    for (miss_rate in MISS_RATES) {
      cat(sprintf(" %g%%", miss_rate*100))
      n_loaded_this_rate <- 0L

      for (sp in seq_len(N_SPLITS)) {
        sp_ckpt <- split_ckpt_file(ds$name, mech, miss_rate, sp)
        sp_data <- load_split_ckpt(sp_ckpt)

        if (!is.null(sp_data)) {
          # ── Load from split checkpoint ──
          ds_results <- c(ds_results, list(sp_data$rows))
          if (!is.null(sp_data$var_rows) && nrow(sp_data$var_rows) > 0)
            ds_var_rows <- c(ds_var_rows, list(sp_data$var_rows))
          if (!is.null(sp_data$fb_row))
            ds_fb_rows  <- c(ds_fb_rows,  list(sp_data$fb_row))
          if (!is.null(sp_data$miss_log_row))
            ds_miss_log <- c(ds_miss_log, list(sp_data$miss_log_row))
          all_time_rows[[length(all_time_rows)+1]] <- data.frame(
            Dataset=ds$name, MissMech=mech, MissRate=miss_rate,
            Split=sp, elapsed_sec=round(sp_data$elapsed_sec, 3),
            from_ckpt=TRUE, stringsAsFactors=FALSE)
          if (!is.null(sp_data$mehas_elapsed) && !is.null(sp_data$fb_row)) {
            all_mehas_time_rows[[length(all_mehas_time_rows)+1]] <- data.frame(
              Dataset=ds$name, MissMech=mech, MissRate=miss_rate, Split=sp,
              n_train=NA_integer_, long_format_rows=NA_integer_,
              mehas_elapsed_sec=round(sp_data$mehas_elapsed, 4),
              fallback=isTRUE(sp_data$fb_row$fb_me_has),
              stringsAsFactors=FALSE)
          }
          n_loaded_this_rate <- n_loaded_this_rate + 1L
          cat("~")   # ~ means loaded from ckpt
          next
        }

        # ── Run this split ──
        sp_seed <- make_split_seed(sp, ds$name, miss_rate, mech)
        set.seed(sp_seed)
        tr_idx  <- caret::createDataPartition(DATASET$event, p=TRAIN_FRAC, list=FALSE)[, 1]
        te_idx  <- setdiff(seq_len(nrow(DATASET)), tr_idx)
        X_tr    <- DATASET[tr_idx, PRED_NAMES, drop=FALSE]
        time_tr <- DATASET[tr_idx, "time"]; evt_tr <- DATASET[tr_idx, "event"]
        X_te    <- DATASET[te_idx, PRED_NAMES, drop=FALSE]
        time_te <- DATASET[te_idx, "time"]; evt_te <- DATASET[te_idx, "event"]

        X_tr_miss_check <- tryCatch(
          induce_missing(X_tr, miss_rate, mech,
                         time_vec=time_tr, event_vec=evt_tr),
          error=function(e) X_tr)
        actual_miss_pct <- round(mean(is.na(X_tr_miss_check)) * 100, 2)
        if (sp == 1) cat(sprintf("[actual=%.1f%%]", actual_miss_pct))

        miss_log_row <- data.frame(
          Dataset=ds$name, MissMech=mech, NominalRate=miss_rate,
          NominalPct=miss_rate*100, ActualMiss=actual_miss_pct/100,
          ActualPct=actual_miss_pct, Split=sp, stringsAsFactors=FALSE)
        ds_miss_log <- c(ds_miss_log, list(miss_log_row))

        split_t0 <- proc.time()
        out <- tryCatch(
          run_one_surv(mech, miss_rate, sp_seed,
                       X_tr, time_tr, evt_tr,
                       X_te, time_te, evt_te),
          error=function(e) {
            message(sprintf("\n  [ERR %s %s %.0f%% sp=%d] %s",
                            ds$name, mech, miss_rate*100, sp,
                            conditionMessage(e)))
            NULL
          })
        split_elapsed <- (proc.time() - split_t0)[["elapsed"]]

        if (!is.null(out)) {
          res            <- out$rows
          res$Dataset    <- ds$name; res$MissMech  <- mech
          res$MissRate   <- miss_rate; res$Split    <- sp
          res$ActualMiss <- actual_miss_pct/100

          vr <- out$var_rows
          if (!is.null(vr) && nrow(vr) > 0) {
            vr$Dataset <- ds$name; vr$Split <- sp
          }

          fbr         <- out$fb_row
          fbr$Dataset <- ds$name; fbr$Split <- sp

          # ── Save split-level checkpoint immediately ──
          save_split_ckpt(
            sp_ckpt,
            rows         = res,
            var_rows     = vr,
            fb_row       = fbr,
            miss_log_row = miss_log_row,
            mehas_elapsed = out$mehas_elapsed,
            elapsed_sec   = split_elapsed)

          ds_results  <- c(ds_results,  list(res))
          if (!is.null(vr) && nrow(vr) > 0)
            ds_var_rows <- c(ds_var_rows, list(vr))
          ds_fb_rows  <- c(ds_fb_rows,   list(fbr))

          all_time_rows[[length(all_time_rows)+1]] <- data.frame(
            Dataset=ds$name, MissMech=mech, MissRate=miss_rate,
            Split=sp, elapsed_sec=round(split_elapsed,3),
            from_ckpt=FALSE, stringsAsFactors=FALSE)

          all_mehas_time_rows[[length(all_mehas_time_rows)+1]] <- data.frame(
            Dataset=ds$name, MissMech=mech, MissRate=miss_rate, Split=sp,
            n_train=length(time_tr),
            long_format_rows=length(time_tr)*length(SURV_NAMES)*M_IMP,
            mehas_elapsed_sec=round(out$mehas_elapsed, 4),
            fallback=isTRUE(fbr$fb_me_has),
            stringsAsFactors=FALSE)
        }
        cat(".")   # . means freshly computed
      }
      if (n_loaded_this_rate > 0)
        cat(sprintf("[%d ckpt]", n_loaded_this_rate))
      cat(" ")
    }
    cat("done\n")
  }

  ds_elapsed <- (proc.time() - ds_t0)[["elapsed"]]
  cat(sprintf("  [Dataset total: %.1f sec / %.1f min]\n",
              ds_elapsed, ds_elapsed/60))

  # ── Save dataset-level summary ckpt (for fast reload next time) ──────────
  saveRDS(list(results  = ds_results,
               var_rows = ds_var_rows,
               fb_rows  = ds_fb_rows,
               miss_log = ds_miss_log), ds_ckpt)
  cat(sprintf("  [DS-CKPT saved: %s]\n", basename(ds_ckpt)))

  all_results  <- c(all_results,  ds_results)
  all_var_rows <- c(all_var_rows, ds_var_rows)
  all_fb_rows  <- c(all_fb_rows,  ds_fb_rows)
  all_miss_log <- c(all_miss_log, ds_miss_log)
}

pipeline_elapsed <- (proc.time() - pipeline_t0)[["elapsed"]]
cat(sprintf("\n=== Done === (%.1f sec / %.1f min)\n",
            pipeline_elapsed, pipeline_elapsed/60))

# ── Save timing ───────────────────────────────────────────────────────────────
if (length(all_time_rows) > 0) {
  time_df   <- do.call(rbind, all_time_rows); rownames(time_df) <- NULL
  total_row <- data.frame(Dataset="__TOTAL_PIPELINE__", MissMech=NA_character_,
                          MissRate=NA_real_, Split=NA_integer_,
                          elapsed_sec=round(pipeline_elapsed,3), stringsAsFactors=FALSE)
  write.csv(rbind(time_df, total_row),
            file.path(OUTPUT_DIR,"Multi_HAS_Survival_compute_time.csv"), row.names=FALSE)
  cat("Saved: Multi_HAS_Survival_compute_time.csv\n")
}
if (length(all_mehas_time_rows) > 0) {
  mehas_time_df <- do.call(rbind, all_mehas_time_rows); rownames(mehas_time_df) <- NULL
  write.csv(mehas_time_df,
            file.path(OUTPUT_DIR,"Multi_HAS_Survival_mehas_timing.csv"), row.names=FALSE)
  cat("Saved: Multi_HAS_Survival_mehas_timing.csv\n")
  cat(sprintf("  ME-HAS mean: %.4f sec | median: %.4f sec | fallbacks: %.1f%%\n",
              mean(mehas_time_df$mehas_elapsed_sec),
              median(mehas_time_df$mehas_elapsed_sec),
              100*mean(mehas_time_df$fallback)))
}

# ── 15. Combine & Save ────────────────────────────────────────────────────────
# safe_rbind: pads any missing columns before combining to prevent
# "numbers of columns do not match" errors when old checkpoints lack new columns
safe_rbind <- function(lst, required_cols=NULL) {
  if (!is.null(required_cols))
    lst <- lapply(lst, pad_df, required_cols=required_cols)
  out <- do.call(rbind, lst)
  rownames(out) <- NULL
  out
}

combined          <- safe_rbind(all_results, RESULT_COLS)
combined$MissRate <- as.numeric(combined$MissRate)
combined$MissMech <- factor(combined$MissMech, levels=MISS_MECHS)
combined$Method   <- factor(combined$Method,   levels=METHOD_ORDER)
write.csv(combined, file.path(OUTPUT_DIR,"Multi_HAS_Survival_results.csv"), row.names=FALSE)
cat("Saved: Multi_HAS_Survival_results.csv\n")

if (length(all_miss_log) > 0) {
  miss_df <- do.call(rbind, all_miss_log); rownames(miss_df) <- NULL
  write.csv(miss_df,
            file.path(OUTPUT_DIR,"Multi_HAS_Survival_missingness_log.csv"), row.names=FALSE)
  cat("Saved: Multi_HAS_Survival_missingness_log.csv\n")
}
if (length(all_fb_rows) > 0) {
  fb_combined <- safe_rbind(all_fb_rows, FB_COLS)
  write.csv(fb_combined,
            file.path(OUTPUT_DIR,"Multi_HAS_Survival_fallbacks.csv"), row.names=FALSE)
  cat("\n╔══════════════════════════════════════════════╗\n")
  cat("║       FALLBACK AUDIT                        ║\n")
  cat("╠══════════════════════════════════════════════╣\n")
  fb_cols   <- c("fb_knn","fb_rfi","fb_missind","fb_mice_na","fb_mice_cart",
                 "fb_mice_se_imp","fb_mice_se_l0","fb_me_has")
  fb_labels <- c("KNN→MeanSI","RFI→MeanSI","MissInd→MeanSI","MICE+NA→stdMICE",
                 "CART→PMM","MICE_SE_imp→reuse","MICE_SE_L0→avg","ME_HAS→avg")
  total_runs <- nrow(fb_combined); all_clean <- TRUE
  for (i in seq_along(fb_cols)) {
    n_fb <- sum(fb_combined[[fb_cols[i]]], na.rm=TRUE)
    pct  <- round(100*n_fb/total_runs, 1)
    flag <- if (n_fb==0) "✓ OK " else "✗ WARN"
    if (n_fb > 0) all_clean <- FALSE
    cat(sprintf("║  %-22s  %s  %4d/%d (%.1f%%)  ║\n",
                fb_labels[i], flag, n_fb, total_runs, pct))
  }
  cat("╠══════════════════════════════════════════════╣\n")
  cat(sprintf("║  %s  ║\n",
              if (all_clean) "ALL METHODS CORRECT — NO FALLBACKS    "
              else            "SOME FALLBACKS — check CSV            "))
  cat("╚══════════════════════════════════════════════╝\n")
  cat("Saved: Multi_HAS_Survival_fallbacks.csv\n")
}

var_combined <- if (length(all_var_rows) > 0) {
  vc <- do.call(rbind, all_var_rows); rownames(vc) <- NULL
  vc$MissRate <- as.numeric(vc$MissRate)
  vc$MissMech <- factor(vc$MissMech, levels=MISS_MECHS); vc
} else NULL
if (!is.null(var_combined)) {
  write.csv(var_combined,
            file.path(OUTPUT_DIR,"Multi_HAS_Survival_variance.csv"), row.names=FALSE)
  cat("Saved: Multi_HAS_Survival_variance.csv\n")
}

# ── 16. Summaries ─────────────────────────────────────────────────────────────
per_ds_summ <- combined %>%
  dplyr::group_by(Dataset, MissMech, MissRate, Method, Classifier) %>%
  dplyr::summarise(
    n_splits    = dplyr::n(),
    Cindex_mean = round(mean(Cindex, na.rm=TRUE), 4),
    Cindex_sd   = round(sd(Cindex,   na.rm=TRUE), 4),
    Uno_C_mean  = round(mean(Uno_C,  na.rm=TRUE), 4),
    Uno_C_sd    = round(sd(Uno_C,    na.rm=TRUE), 4),
    .groups="drop")
write.csv(per_ds_summ,
          file.path(OUTPUT_DIR,"Multi_HAS_Survival_per_dataset_summary.csv"), row.names=FALSE)

grand_summ <- combined %>%
  dplyr::group_by(MissMech, MissRate, Method, Classifier) %>%
  dplyr::summarise(
    n_obs       = dplyr::n(),
    Cindex_mean = round(mean(Cindex, na.rm=TRUE), 4),
    Cindex_sd   = round(sd(Cindex,   na.rm=TRUE), 4),
    Uno_C_mean  = round(mean(Uno_C,  na.rm=TRUE), 4),
    Uno_C_sd    = round(sd(Uno_C,    na.rm=TRUE), 4),
    .groups="drop")
write.csv(grand_summ,
          file.path(OUTPUT_DIR,"Multi_HAS_Survival_grand_summary.csv"), row.names=FALSE)
cat("Saved: summaries\n")

# ── 17. Wilcoxon Signed-Rank Tests ────────────────────────────────────────────
# Bonferroni correction over N_BONF_TOTAL = 280 simultaneous tests.
# Family: 28 competitor method–classifier combinations × 2 metrics × 5 rates,
# computed separately per missing-data mechanism (MAR and MCAR).

IMPUTE_METHODS <- c("MeanSI","KNN","RFI","MissInd","MIE_PMM_AVG","MIE_CART_AVG","MICE_SE")

ds_level <- combined %>%
  dplyr::group_by(Dataset, MissMech, MissRate, Method, Classifier) %>%
  dplyr::summarise(Cindex_ds=mean(Cindex, na.rm=TRUE),
                   Uno_C_ds =mean(Uno_C,  na.rm=TRUE), .groups="drop")

RANK_METHODS <- c("MeanSI","KNN","RFI","MissInd","MIE_PMM_AVG","MIE_CART_AVG","MICE_SE","ME_HAS")
wilcox_rows  <- list()

champ_ds <- ds_level %>%
  dplyr::filter(Method=="ME_HAS", Classifier=="All") %>%
  dplyr::select(Dataset, MissMech, MissRate,
                Cindex_ch=Cindex_ds, Uno_C_ch=Uno_C_ds)

comp_combos <- ds_level %>%
  dplyr::filter(Method %in% IMPUTE_METHODS) %>%
  dplyr::distinct(Method, Classifier) %>%
  dplyr::mutate(Method=factor(Method, levels=METHOD_ORDER)) %>%
  dplyr::arrange(Method, Classifier)

# Bonferroni correction applied per mechanism separately
# (MAR and MCAR reported in separate tables → separate family of tests)
# Per mechanism: 28 combos × 2 metrics × 5 rates = 280
n_combos        <- nrow(comp_combos)
n_metrics       <- 2L
n_rates         <- length(MISS_RATES)
N_BONF_PER_MECH <- n_combos * n_metrics * n_rates
cat(sprintf("Bonferroni: %d combos × %d metrics × %d rates = %d per mechanism\n",
            n_combos, n_metrics, n_rates, N_BONF_PER_MECH))

for (ci in seq_len(n_combos)) {
  meth <- as.character(comp_combos$Method[ci])
  clf  <- as.character(comp_combos$Classifier[ci])
  comp_ds <- ds_level %>%
    dplyr::filter(Method==meth, Classifier==clf) %>%
    dplyr::select(Dataset, MissMech, MissRate,
                  Cindex_comp=Cindex_ds, Uno_C_comp=Uno_C_ds)
  if (nrow(comp_ds) == 0) next
  paired_all <- dplyr::inner_join(champ_ds, comp_ds,
                                  by=c("Dataset","MissMech","MissRate"))
  for (mech in MISS_MECHS)
    for (mr in MISS_RATES) {
      paired <- paired_all %>%
        dplyr::filter(MissMech==mech, abs(MissRate-mr)<1e-6)
      if (nrow(paired) < 2) next
      for (metric in c("Cindex","Uno_C")) {
        champ_col <- if (metric=="Cindex") paired$Cindex_ch else paired$Uno_C_ch
        comp_col  <- if (metric=="Cindex") paired$Cindex_comp else paired$Uno_C_comp
        diff_vec  <- champ_col - comp_col
        n <- sum(!is.na(diff_vec)); if (n < 2) next
        wt <- tryCatch(wilcox.test(champ_col, comp_col, paired=TRUE, exact=FALSE),
                       error=function(e) NULL)
        if (is.null(wt)) next
        # Bonferroni: correct within each mechanism separately
        p_adj  <- round(min(wt$p.value * N_BONF_PER_MECH, 1), 5)
        n_wins <- sum(diff_vec > 0, na.rm=TRUE)
        n_loss <- sum(diff_vec < 0, na.rm=TRUE)
        r_rbc  <- round((n_wins - n_loss) / n, 3)
        wilcox_rows[[length(wilcox_rows)+1]] <- data.frame(
          Champion   = "ME_HAS",
          Metric     = metric,
          MissMech   = mech,
          MissRate   = mr,
          MissRate_pct = paste0(round(mr*100), "%"),
          Competitor = meth,
          Classifier = clf,
          n_pairs    = n,
          W          = round(wt$statistic, 1),
          p_value    = round(wt$p.value, 5),
          p_adj_Bonf = p_adj,
          Signif     = dplyr::case_when(
            p_adj<0.001 ~ "***", p_adj<0.01 ~ "**",
            p_adj<0.05  ~ "*",   TRUE        ~ "ns"),
          r_rank_biserial = r_rbc,
          ES_interp  = dplyr::case_when(
            abs(r_rbc)>=0.3 ~ "large",
            abs(r_rbc)>=0.1 ~ "medium", TRUE ~ "small"),
          mean_diff       = round(mean(diff_vec, na.rm=TRUE), 4),
          sd_diff         = round(sd(diff_vec,   na.rm=TRUE), 4),
          Champion_wins   = n_wins,
          Ties            = sum(diff_vec==0, na.rm=TRUE),
          Champion_losses = n_loss,
          stringsAsFactors=FALSE)
      }
    }
}
wilcox_df <- do.call(rbind, wilcox_rows); rownames(wilcox_df) <- NULL
write.csv(wilcox_df,
          file.path(OUTPUT_DIR,"Multi_HAS_Survival_wilcoxon.csv"), row.names=FALSE)
cat("Saved: Multi_HAS_Survival_wilcoxon.csv\n")

# ── 18. LaTeX Helpers ─────────────────────────────────────────────────────────
`%.%` <- function(a,b) paste0(a,b)

method_tex <- function(meth)
  switch(as.character(meth),
         ME_HAS       = "\\textbf{ME-HAS (proposed)}",
         MICE_SE      = "MICE-SE (Stacking)",
         MIE_PMM_AVG  = "MIE-PMM-Avg",
         MIE_CART_AVG = "MIE-CART-Avg",
         MeanSI       = "Mean SI",
         MissInd      = "Missing Indicator",
         Complete     = "Complete",
         as.character(meth))

fmt_cell <- function(mn, sd, bold=FALSE) {
  if (is.na(mn)) return("{--}")
  sd <- if (is.na(sd)) 0 else sd
  if (bold) sprintf("$\\mathbf{%.3f}{\\pm}\\mathbf{%.3f}$", mn, sd)
  else      sprintf("$%.3f{\\pm}%.3f$", mn, sd)
}

latex_preamble <- function(fs=10)
  c(sprintf("\\documentclass[%dpt]{article}", fs),
    "\\usepackage[utf8]{inputenc}", "\\usepackage[T1]{fontenc}",
    "\\usepackage{booktabs,geometry,amsmath,graphicx,bm}",
    "\\usepackage[table]{xcolor}",
    "\\geometry{margin=1.2cm,landscape,a3paper}",
    "\\begin{document}", "")

pivot_wide_ms <- function(df, metric) {
  mc <- paste0(metric,"_mean"); sc <- paste0(metric,"_sd")
  df2 <- df %>%
    dplyr::mutate(col_key=dplyr::case_when(
      abs(MissRate)<1e-6 ~ "COMP",
      TRUE ~ paste0(as.character(MissMech),
                    sprintf("%02d", as.integer(round(MissRate*100)))))) %>%
    dplyr::group_by(Method, Classifier, col_key) %>%
    dplyr::summarise(!!mc := mean(.data[[mc]], na.rm=TRUE),
                     !!sc := mean(.data[[sc]], na.rm=TRUE), .groups="drop")
  wm <- df2 %>%
    dplyr::select(Method, Classifier, col_key, val=dplyr::all_of(mc)) %>%
    tidyr::pivot_wider(names_from=col_key, values_from=val)
  ws <- df2 %>%
    dplyr::select(Method, Classifier, col_key, val=dplyr::all_of(sc)) %>%
    tidyr::pivot_wider(names_from=col_key, values_from=val, names_prefix="SD_")
  dplyr::left_join(wm, ws, by=c("Method","Classifier"))
}

build_latex_table <- function(wide, metric, caption_str, label_str,
                              target_mech="MAR") {
  mech_keys <- paste0(target_mech,
                      sprintf("%02d", as.integer(MISS_RATES*100)))
  present   <- intersect(c("COMP", mech_keys), names(wide))
  nc  <- sum("COMP" %in% present)
  nm2 <- sum(mech_keys %in% present)
  nd  <- nc + nm2
  wide <- wide %>%
    dplyr::mutate(Method=factor(Method, levels=METHOD_ORDER)) %>%
    dplyr::arrange(Method, Classifier)
  best_row <- sapply(present, function(k) {
    vals <- unlist(wide[[k]])
    if (all(is.na(vals))) return(NA_integer_)
    which(vals==max(vals, na.rm=TRUE))[1]
  })
  lines <- c("\\begin{table}[!ht]", "  \\centering",
             sprintf("  \\caption{%s}", caption_str),
             sprintf("  \\label{%s}", label_str),
             "  \\resizebox{\\linewidth}{!}{%",
             sprintf("  \\begin{tabular}{@{}ll%s@{}}", strrep("c", nd)),
             "  \\toprule")
  sc <- c("\\textbf{Method}", "\\textbf{Model}")
  if (nc  > 0) sc <- c(sc, sprintf("\\multicolumn{%d}{c}{\\textbf{Complete}}", nc))
  if (nm2 > 0) sc <- c(sc, sprintf("\\multicolumn{%d}{c}{\\textbf{%s}}", nm2, target_mech))
  lines <- c(lines, sprintf("  %s \\\\", paste(sc, collapse=" & ")))
  cs2 <- 3L; cmi <- character(0)
  if (nc  > 0) { cmi <- c(cmi, sprintf("\\cmidrule(lr){%d-%d}", cs2, cs2+nc-1)); cs2 <- cs2+nc }
  if (nm2 > 0)   cmi <- c(cmi, sprintf("\\cmidrule(lr){%d-%d}", cs2, cs2+nm2-1))
  if (length(cmi) > 0) lines <- c(lines, paste("  ", paste(cmi, collapse=" ")))
  rl  <- paste0(as.integer(MISS_RATES*100), "\\%")
  sub <- c("", "")
  if (nc  > 0) sub <- c(sub, "--")
  if (nm2 > 0) sub <- c(sub, rl[seq_len(nm2)])
  lines <- c(lines, sprintf("  %s \\\\", paste(sub, collapse=" & ")),
             "  \\midrule")
  prev <- ""
  for (i in seq_len(nrow(wide))) {
    mth <- as.character(wide$Method[i])
    if (mth != prev && i > 1) lines <- c(lines, "  \\midrule")
    prev <- mth
    cells <- sapply(seq_along(present), function(ci) {
      k  <- present[ci]
      mn <- unlist(wide[[k]])[i]
      sd <- unlist(wide[[paste0("SD_", k)]])[i]
      fmt_cell(mn, sd, bold=(!is.na(best_row[ci]) && best_row[ci]==i))
    })
    lines <- c(lines, sprintf("  %s & %s & %s \\\\",
                              method_tex(mth), wide$Classifier[i],
                              paste(cells, collapse=" & ")))
  }
  lines <- c(lines, "  \\bottomrule", "  \\end{tabular}%",
             "  }", "\\end{table}", "")
  paste(lines, collapse="\n")
}

make_grand_table <- function(gs, metric="Cindex", target_mech="MAR") {
  gs_m <- gs %>%
    dplyr::filter(abs(MissRate)<1e-6 | MissMech==target_mech)
  mech_desc <- if (target_mech=="MAR")
    "MAR mechanism: logistic model, $\\beta=\\log(9)$, OR$\\approx$9 per SD \\citep{WhiteRoyston2009}."
  else
    "MCAR mechanism: columns masked completely at random."
  metric_desc <- if (metric=="Cindex")
    "Harrell C-index (concordance; higher is better) \\citep{Harrell1996}"
  else
    paste0("Uno's C-statistic (IPCW-weighted concordance; higher is better) \\citep{Uno2011}.",
           " Evaluated via \\texttt{pec::cindex()} at $\\tau_{0.75}$",
           " (75th percentile of observed event times).")
  cap <- sprintf(
    paste0("Grand mean %s ($\\text{mean}\\pm\\text{SD}$) averaged over %d",
           " right-censored survival benchmark datasets and %d stratified",
           " train--test splits (70/30). \\textit{Complete}: no missingness",
           " imposed. \\textit{ME-HAS} (proposed): Mixed-Effects Hierarchical",
           " Aware Stacking. \\textit{Missing Indicator} (MissInd): binary",
           " missingness flags appended to mean-imputed data",
           " \\citep{Jones1996,Sperrin2020}. %s",
           " MICE uses PMM with Nelson--Aalen predictor",
           " \\citep{WhiteRoyston2009,vanBuurenGroothuis2011}."),
    metric_desc, length(DATASET_REGISTRY), N_SPLITS, mech_desc)
  build_latex_table(
    pivot_wide_ms(gs_m, metric), metric, cap,
    sprintf("tab:grand_%s_%s", tolower(metric), tolower(target_mech)),
    target_mech=target_mech)
}

make_kendall_rank_table <- function(metric="Cindex", target_mech="MAR",
                                    target_rate=0.30) {
  score_col   <- if (metric=="Cindex") "Cindex_ds" else "Uno_C_ds"
  imp_methods <- setdiff(RANK_METHODS, "ME_HAS")
  imp_sub <- ds_level %>%
    dplyr::filter(MissMech==target_mech,
                  abs(MissRate-target_rate)<1e-6,
                  Method %in% imp_methods,
                  Classifier %in% SURV_NAMES) %>%
    dplyr::mutate(RowKey=paste0(Method,"___",Classifier)) %>%
    dplyr::select(Dataset, RowKey, val=dplyr::all_of(score_col))
  has_sub <- ds_level %>%
    dplyr::filter(MissMech==target_mech,
                  abs(MissRate-target_rate)<1e-6,
                  Method=="ME_HAS", Classifier=="All") %>%
    dplyr::mutate(RowKey="ME_HAS___All") %>%
    dplyr::select(Dataset, RowKey, val=dplyr::all_of(score_col))
  sub <- dplyr::bind_rows(imp_sub, has_sub)
  if (nrow(sub) < 2) return("")
  wide <- sub %>% tidyr::pivot_wider(names_from=Dataset, values_from=val)
  row_order <- c(unlist(lapply(imp_methods,
                               function(m) paste0(m,"___",SURV_NAMES))),
                 "ME_HAS___All")
  row_order <- row_order[row_order %in% wide$RowKey]
  wide      <- wide[match(row_order, wide$RowKey), ]
  datasets  <- setdiff(names(wide), "RowKey")
  score_mat <- wide %>%
    dplyr::select(dplyr::all_of(datasets)) %>% as.matrix()
  rownames(score_mat) <- wide$RowKey
  rank_mat <- apply(score_mat, 2, function(x) {
    if (sum(!is.na(x)) < 2) return(rep(NA_real_, length(x)))
    rank(-x, ties.method="average", na.last="keep")
  })
  rownames(rank_mat) <- wide$RowKey
  valid_cols <- colSums(!is.na(rank_mat)) >= 2
  valid_rows <- rowSums(!is.na(rank_mat[, valid_cols, drop=FALSE])) >= 2
  rm_clean   <- rank_mat[valid_rows, valid_cols, drop=FALSE]
  n_obj <- nrow(rm_clean); m_judges <- ncol(rm_clean)
  W_val <- chi_val <- p_val <- df_val <- NA_real_; signif_str <- ""
  if (n_obj >= 2 && m_judges >= 2) {
    rs      <- rowSums(rm_clean, na.rm=TRUE)
    mean_rs <- m_judges*(n_obj+1)/2
    S       <- sum((rs-mean_rs)^2)
    W_val   <- 12*S / (m_judges^2*(n_obj^3-n_obj))
    chi_val <- m_judges*(n_obj-1)*W_val
    df_val  <- n_obj-1
    p_val   <- pchisq(chi_val, df=df_val, lower.tail=FALSE)
    signif_str <- dplyr::case_when(p_val<0.001~"***", p_val<0.01~"**",
                                   p_val<0.05~"*",    TRUE~"ns")
  }
  mean_rank_per_row    <- rowMeans(rank_mat, na.rm=TRUE)
  valid_keys_unsorted  <- row_order[row_order %in% names(mean_rank_per_row)]
  row_order <- valid_keys_unsorted[order(mean_rank_per_row[valid_keys_unsorted])]
  row_label_tex <- function(key) {
    parts <- strsplit(key, "___")[[1]]; meth <- parts[1]; clf <- parts[2]
    m_lab <- switch(meth,
                    MeanSI       = "Mean SI",
                    KNN          = "KNN (SI)",
                    RFI          = "RFI (SI)",
                    MissInd      = "Missing Indicator",
                    MIE_PMM_AVG  = "MIE-PMM-Avg",
                    MIE_CART_AVG = "MIE-CART-Avg",
                    MICE_SE      = "MICE-SE",
                    ME_HAS       = "\\textbf{ME-HAS (proposed)}",
                    meth)
    if (clf=="All") m_lab else sprintf("%s --- %s", m_lab, clf)
  }
  fmt_note <- if (metric=="Cindex") "higher C-index $=$ better"
  else                   "higher Uno's C $=$ better"
  ds_hdrs <- sapply(datasets,
                    function(d) sprintf("\\textbf{\\rotatebox{45}{%s}}", d))
  col_hdr <- c("\\textbf{Method}", ds_hdrs, "\\textbf{Mean}")
  ncols   <- length(col_hdr)
  lines <- c(
    "\\begin{table}[!ht]", "  \\centering",
    sprintf(paste0(
      "  \\caption{Ranks of all method--classifier combinations across",
      " %d datasets (\\emph{%s}, %s, %.0f\\%% %s; %s),",
      " sorted by mean rank (ascending): top row = overall best.",
      " Rank~1~=~best per dataset; \\textbf{bold}~=~column-best.",
      " \\textbf{Mean}: average rank (lower is better).",
      " Kendall's $W$ (footer) measures rank consistency.",
      " $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$, ns~=~not significant.}"),
      length(datasets), metric, target_mech,
      target_rate*100, target_mech, fmt_note),
    sprintf("  \\label{tab:rank_%s_%s_%02d}",
            tolower(metric), tolower(target_mech),
            as.integer(target_rate*100)),
    "  \\resizebox{\\linewidth}{!}{%",
    sprintf("  \\begin{tabular}{@{}l%s@{}}", strrep("c", ncols-1)),
    "  \\toprule",
    paste(col_hdr, collapse=" & ") %.% " \\\\",
    "  \\midrule")
  valid_keys   <- row_order[row_order %in% rownames(rank_mat)]
  best_per_col <- apply(rank_mat[valid_keys, datasets, drop=FALSE], 2,
                        function(x) suppressWarnings(min(x, na.rm=TRUE)))
  best_mean    <- suppressWarnings(min(mean_rank_per_row[valid_keys], na.rm=TRUE))
  for (key in row_order) {
    if (!key %in% rownames(rank_mat)) next
    row_vals <- rank_mat[key, datasets]
    mean_val <- mean_rank_per_row[key]
    cells <- mapply(function(v, col_best) {
      if (is.na(v)||!is.finite(v)) return("--")
      s <- sprintf("%.1f", v)
      if (is.finite(col_best) && abs(v-col_best)<1e-8)
        paste0("\\textbf{", s, "}") else s
    }, row_vals, best_per_col[datasets], SIMPLIFY=TRUE)
    mean_cell <- if (is.finite(mean_val)) {
      s <- sprintf("%.2f", mean_val)
      if (is.finite(best_mean) && abs(mean_val-best_mean)<1e-8)
        paste0("\\textbf{", s, "}") else s
    } else "--"
    lines <- c(lines,
               paste(c(row_label_tex(key), cells, mean_cell),
                     collapse=" & ") %.% " \\\\")
  }
  lines <- c(lines, "  \\midrule")
  w_str   <- if (!is.na(W_val))   sprintf("%.4f", W_val)           else "--"
  chi_str <- if (!is.na(chi_val)) sprintf("%.3f", chi_val)         else "--"
  p_str   <- if (!is.na(p_val))   sprintf("%.5f", p_val)           else "--"
  df_str  <- if (!is.na(df_val))  as.character(as.integer(df_val)) else "?"
  w_content <- sprintf(
    "$W = %s$,\\ $\\chi^2_{%s} = %s$,\\ $p = %s$\\ %s",
    w_str, df_str, chi_str, p_str, signif_str)
  lines <- c(lines,
             sprintf("  Kendall's $W$ & \\multicolumn{%d}{c}{%s} \\\\",
                     length(datasets)+1L, w_content),
             "  \\bottomrule", "  \\end{tabular}%",
             "  }", "\\end{table}", "")
  paste(lines, collapse="\n")
}

make_varcomp_table <- function(target_mech="MAR") {
  if (is.null(var_combined)||nrow(var_combined)==0) return("")
  vc <- var_combined %>%
    dplyr::filter(Classifier=="ME_HAS", MissMech==target_mech) %>%
    dplyr::group_by(Dataset, MissRate) %>%
    dplyr::summarise(slope_mean =mean(var_slope_clf,    na.rm=TRUE),
                     interc_mean=mean(var_intercept_imp, na.rm=TRUE),
                     .groups="drop")
  if (nrow(vc)==0) return("")
  mr_pcts  <- sort(unique(vc$MissRate))
  datasets <- unique(vc$Dataset)
  fmt_varcomp_cell <- function(s, ic) {
    sv <- if (is.na(s)||!is.finite(s))  "---" else sprintf("%.4f", s)
    iv <- if (is.na(ic)||!is.finite(ic)) "---" else sprintf("%.4f", ic)
    sprintf("$\\shortstack[c]{%s \\\\\\\\ %s}$", sv, iv)
  }
  mr_hdrs <- sprintf("\\textbf{%d\\%%}", as.integer(mr_pcts*100))
  col_hdr <- c("\\textbf{Dataset}", mr_hdrs, "\\textbf{Mean}")
  ncols   <- length(col_hdr)
  lines <- c(
    "\\begin{table}[!ht]", "  \\centering",
    sprintf(paste0(
      "  \\caption{Mean ME-HAS variance components averaged over %d splits,",
      " by dataset and missingness proportion (%s). Each cell:",
      " \\emph{top} = $\\hat{\\sigma}^2_{\\mathrm{slope}}$",
      " ($\\mathrm{lp}\\mid\\texttt{clf\\_id}$; between-classifier",
      " heterogeneity); \\emph{bottom} =",
      " $\\hat{\\sigma}^2_{\\mathrm{intercept}}$",
      " ($1\\mid\\texttt{imp\\_id}$; between-imputation baseline risk,",
      " analogous to Rubin's $B$ \\citep{Rubin1987}).",
      " Extracted via \\texttt{coxme::VarCorr} \\citep{Therneau2012}.}"),
      N_SPLITS, target_mech),
    sprintf("  \\label{tab:varcomp_%s}", tolower(target_mech)),
    "  \\renewcommand{\\arraystretch}{1.4}",
    "  \\resizebox{\\linewidth}{!}{%",
    sprintf("  \\begin{tabular}{@{}l%s@{}}", strrep("c", ncols-1L)),
    "  \\toprule",
    paste(col_hdr, collapse=" & ") %.% " \\\\",
    sprintf(paste0("  \\multicolumn{%d}{l}{\\footnotesize",
                   " top: $\\hat{\\sigma}^2_{\\mathrm{slope}}$",
                   " $\\mid$ bottom:",
                   " $\\hat{\\sigma}^2_{\\mathrm{intercept}}$} \\\\"), ncols),
    "  \\midrule")
  ds_slope_means  <- numeric(length(datasets))
  ds_interc_means <- numeric(length(datasets))
  for (di in seq_along(datasets)) {
    ds_sub <- vc %>% dplyr::filter(Dataset==datasets[di])
    cells <- sapply(mr_pcts, function(mr) {
      row <- ds_sub %>% dplyr::filter(abs(MissRate-mr)<1e-6)
      if (nrow(row)==0) return("--")
      fmt_varcomp_cell(row$slope_mean[1], row$interc_mean[1])
    })
    ms  <- mean(ds_sub$slope_mean,  na.rm=TRUE)
    mic <- mean(ds_sub$interc_mean, na.rm=TRUE)
    ds_slope_means[di]  <- ms
    ds_interc_means[di] <- mic
    lines <- c(lines,
               paste(c(sprintf("\\textit{%s}", datasets[di]),
                       cells, fmt_varcomp_cell(ms, mic)),
                     collapse=" & ") %.% " \\\\")
  }
  lines <- c(lines, "  \\midrule")
  grand_cells <- sapply(mr_pcts, function(mr) {
    sub <- vc %>% dplyr::filter(abs(MissRate-mr)<1e-6)
    fmt_varcomp_cell(mean(sub$slope_mean, na.rm=TRUE),
                     mean(sub$interc_mean, na.rm=TRUE))
  })
  grand_mean_cell <- fmt_varcomp_cell(
    mean(ds_slope_means,  na.rm=TRUE),
    mean(ds_interc_means, na.rm=TRUE))
  lines <- c(lines,
             paste(c("\\textbf{Grand mean}", grand_cells, grand_mean_cell),
                   collapse=" & ") %.% " \\\\",
             "  \\bottomrule", "  \\end{tabular}%", "  }",
             "  \\renewcommand{\\arraystretch}{1}", "\\end{table}", "")
  paste(lines, collapse="\n")
}

# ── Per-mechanism LaTeX output ────────────────────────────────────────────────
for (mech in MISS_MECHS) {
  grand_tex <- file.path(OUTPUT_DIR,
                         sprintf("Multi_HAS_Survival_grand_tables_%s.tex", mech))
  gt_lines  <- latex_preamble(10)
  for (met in c("Cindex","Uno_C"))
    gt_lines <- c(gt_lines, make_grand_table(grand_summ, met, target_mech=mech))
  gt_lines <- c(gt_lines, "\\end{document}")
  writeLines(gt_lines, grand_tex)
  cat(sprintf("Saved: %s\n", basename(grand_tex)))

  vc_tex   <- file.path(OUTPUT_DIR,
                        sprintf("Multi_HAS_Survival_varcomp_table_%s.tex", mech))
  vc_lines <- c(latex_preamble(10), make_varcomp_table(mech), "\\end{document}")
  writeLines(vc_lines, vc_tex)
  cat(sprintf("Saved: %s\n", basename(vc_tex)))

  for (met in c("Cindex","Uno_C")) {
    rank_tex <- file.path(OUTPUT_DIR,
                          sprintf("Multi_HAS_Survival_rank_tables_%s_%s.tex", mech, met))
    rk_lines <- c(latex_preamble(9),
                  sprintf("%% Rank Tables — %s, %s\n", mech, met))
    for (mr in MISS_RATES) {
      tbl <- make_kendall_rank_table(met, mech, mr)
      if (nchar(tbl)>0) {
        rk_lines <- c(rk_lines, tbl)
        cat(sprintf("  rank table: %s %s %.0f%%\n", met, mech, mr*100))
      }
    }
    rk_lines <- c(rk_lines, "\\end{document}")
    writeLines(rk_lines, rank_tex)
    cat(sprintf("Saved: %s\n", basename(rank_tex)))
  }
}

# ── 19. Figures ───────────────────────────────────────────────────────────────
# MissInd added: colour, linetype, and label defined alongside other methods
colour_pal <- c(
  Complete     = "#9E9E9E",
  MeanSI       = "#795548",
  KNN          = "#FF9800",
  RFI          = "#8BC34A",
  MissInd      = "#FF5722",   # deep orange — distinct from MeanSI brown
  MIE_PMM_AVG  = "#2196F3",
  MIE_CART_AVG = "#673AB7",
  MICE_SE      = "#00BCD4",
  ME_HAS       = "#E53935"
)
ltype_pal <- c(
  Complete     = "dotted",
  MeanSI       = "solid",
  KNN          = "solid",
  RFI          = "solid",
  MissInd      = "longdash",  # distinguishable from other SI methods
  MIE_PMM_AVG  = "dashed",
  MIE_CART_AVG = "dashed",
  MICE_SE      = "twodash",
  ME_HAS       = "solid"
)
surv_shape_pal <- setNames(c(16L, 17L, 15L, 18L), SURV_NAMES)
METHOD_LABELS  <- c(
  Complete     = "Complete",
  MeanSI       = "Mean (SI)",
  KNN          = "KNN (SI)",
  RFI          = "RFI (SI)",
  MissInd      = "Miss. Indicator (SI)",
  MIE_PMM_AVG  = "MIE-PMM-Avg",
  MIE_CART_AVG = "MIE-CART-Avg",
  MICE_SE      = "MICE-SE",
  ME_HAS       = "ME-HAS (proposed)"
)

theme_paper <- function(base=9)
  theme_bw(base_size=base) +
  theme(panel.grid.minor   = element_blank(),
        panel.grid.major   = element_line(colour="grey88", linewidth=0.3),
        strip.background   = element_rect(fill="white", colour="grey40"),
        strip.text         = element_text(face="bold", size=rel(0.9)),
        legend.position    = "bottom",
        legend.title       = element_blank(),
        legend.text        = element_text(size=rel(0.85)),
        plot.title         = element_text(face="bold", size=rel(1.0), hjust=0),
        axis.title         = element_text(size=rel(0.95)))

for (mech in MISS_MECHS) {
  cat(sprintf("\n=== Figures for %s ===\n", mech))

  clf_raw <- combined %>%
    dplyr::filter(Method %in% METHOD_ORDER, MissMech==mech) %>%
    dplyr::group_by(MissRate, Method, Classifier) %>%
    dplyr::summarise(Mean_Cindex = mean(Cindex, na.rm=TRUE),
                     SD_Cindex   = sd(Cindex,   na.rm=TRUE),
                     Mean_Uno_C  = mean(Uno_C,  na.rm=TRUE),
                     SD_Uno_C    = sd(Uno_C,    na.rm=TRUE),
                     .groups="drop")

  mehas_raw <- clf_raw %>% dplyr::filter(Method=="ME_HAS", Classifier=="All")
  mehas_rep <- purrr::map_dfr(SURV_NAMES, ~dplyr::mutate(mehas_raw, Classifier=.x))

  plot_clf <- dplyr::bind_rows(
    clf_raw %>% dplyr::filter(Method!="ME_HAS"),
    mehas_rep) %>%
    dplyr::mutate(
      Method      = factor(Method, levels=METHOD_ORDER),
      MethodLabel = factor(dplyr::recode(as.character(Method), !!!METHOD_LABELS),
                           levels=unname(METHOD_LABELS)),
      ClfLabel    = factor(Classifier, levels=SURV_NAMES))

  make_traj_plot <- function(y_var, y_lab, fig_title, caption_txt) {
    ggplot(plot_clf %>% dplyr::filter(!is.na(MethodLabel), !is.na(ClfLabel)),
           aes(x=MissRate*100, y=.data[[y_var]],
               colour=MethodLabel, linetype=MethodLabel, shape=ClfLabel,
               group=interaction(MethodLabel, ClfLabel))) +
      { if (length(MISS_RATES)>1) geom_line(linewidth=0.65) else geom_blank() } +
      geom_point(size=1.6) +
      { if (length(MISS_RATES)>1)
        geom_line(
          data=plot_clf %>%
            dplyr::filter(Method=="ME_HAS", !is.na(ClfLabel)),
          aes(x=MissRate*100, y=.data[[y_var]],
              group=interaction(MethodLabel, ClfLabel)),
          colour="#E53935", linewidth=2.0, linetype="solid",
          inherit.aes=FALSE)
        else geom_blank() } +
      facet_wrap(~ClfLabel, nrow=1) +
      scale_colour_manual(values=setNames(colour_pal,   METHOD_LABELS)) +
      scale_linetype_manual(values=setNames(ltype_pal,  METHOD_LABELS)) +
      scale_shape_manual(values=surv_shape_pal) +
      scale_x_continuous(breaks=MISS_RATES*100,
                         labels=paste0(MISS_RATES*100, "%")) +
      labs(title=fig_title, caption=caption_txt,
           x="Missingness Rate (%)", y=y_lab) +
      theme_paper() +
      guides(colour   = guide_legend(nrow=3,
                                     override.aes=list(linewidth=1.2)),
             linetype = guide_legend(nrow=3),
             shape    = guide_legend(nrow=1))
  }

  # Fig A: Grand mean C-index
  pA <- make_traj_plot(
    "Mean_Cindex",
    "Mean Harrell C-index",
    sprintf("Grand Mean C-index Trajectories Under %s", mech),
    sprintf("Grand mean over %d benchmark datasets. MissInd = Missing Indicator.",
            length(unique(combined$Dataset))))
  ggsave(file.path(OUTPUT_DIR, sprintf("figA_grand_cindex_%s.png", mech)),
         plot=pA, width=16, height=5, units="in", dpi=900, bg="white")
  ggsave(file.path(OUTPUT_DIR, sprintf("figA_grand_cindex_%s.pdf", mech)),
         plot=pA, width=16, height=5, units="in", device=cairo_pdf)
  cat(sprintf("Saved: figA_grand_cindex_%s (.png + .pdf)\n", mech))

  # Fig A2: Grand mean Uno's C
  pA2 <- make_traj_plot(
    "Mean_Uno_C",
    "Mean Uno's C-statistic (IPCW)",
    sprintf("Grand Mean Uno's C-statistic Trajectories Under %s", mech),
    sprintf("Grand mean over %d benchmark datasets. IPCW-weighted, tau=Q75 event times.",
            length(unique(combined$Dataset))))
  ggsave(file.path(OUTPUT_DIR, sprintf("figA2_grand_uno_c_%s.png", mech)),
         plot=pA2, width=16, height=5, units="in", dpi=900, bg="white")
  ggsave(file.path(OUTPUT_DIR, sprintf("figA2_grand_uno_c_%s.pdf", mech)),
         plot=pA2, width=16, height=5, units="in", device=cairo_pdf)
  cat(sprintf("Saved: figA2_grand_uno_c_%s (.png + .pdf)\n", mech))

  # Fig B: Per-dataset C-index
  clf_ds_raw <- combined %>%
    dplyr::filter(Method %in% METHOD_ORDER, MissMech==mech) %>%
    dplyr::group_by(MissRate, Method, Classifier, Dataset) %>%
    dplyr::summarise(Mean_Cindex=mean(Cindex, na.rm=TRUE), .groups="drop")
  mehas_ds_raw <- clf_ds_raw %>%
    dplyr::filter(Method=="ME_HAS", Classifier=="All")
  mehas_ds_rep <- purrr::map_dfr(SURV_NAMES,
                                 ~dplyr::mutate(mehas_ds_raw, Classifier=.x))
  plot_clf_ds <- dplyr::bind_rows(
    clf_ds_raw %>% dplyr::filter(Method!="ME_HAS"),
    mehas_ds_rep) %>%
    dplyr::mutate(
      Method      = factor(Method, levels=METHOD_ORDER),
      MethodLabel = factor(dplyr::recode(as.character(Method), !!!METHOD_LABELS),
                           levels=unname(METHOD_LABELS)),
      ClfLabel    = factor(Classifier, levels=SURV_NAMES))

  pB <- ggplot(
    plot_clf_ds %>% dplyr::filter(!is.na(MethodLabel), !is.na(ClfLabel)),
    aes(x=MissRate*100, y=Mean_Cindex,
        colour=MethodLabel, linetype=MethodLabel, shape=ClfLabel,
        group=interaction(MethodLabel, ClfLabel))) +
    { if (length(MISS_RATES)>1) geom_line(linewidth=0.45, alpha=0.7)
      else geom_blank() } +
    geom_point(size=1.4) +
    { if (length(MISS_RATES)>1)
      geom_line(
        data=plot_clf_ds %>%
          dplyr::filter(Method=="ME_HAS", !is.na(ClfLabel)),
        aes(x=MissRate*100, y=Mean_Cindex,
            group=interaction(MethodLabel, ClfLabel)),
        colour="#E53935", linewidth=2.0, linetype="solid",
        inherit.aes=FALSE)
      else geom_blank() } +
    facet_wrap(~Dataset, nrow=3, ncol=3, scales="free_y") +
    scale_colour_manual(values=setNames(colour_pal,   METHOD_LABELS)) +
    scale_linetype_manual(values=setNames(ltype_pal,  METHOD_LABELS)) +
    scale_shape_manual(values=surv_shape_pal) +
    scale_x_continuous(breaks=MISS_RATES*100,
                       labels=paste0(MISS_RATES*100, "%")) +
    labs(title   = sprintf("Dataset-Level C-index Trajectories Under %s", mech),
         caption = "Colour: imputation method; shape: base survival model.",
         x="Missingness Rate (%)", y="Mean C-index per Dataset") +
    theme_paper(base=8) +
    theme(axis.text.x=element_text(angle=45, hjust=1, size=7)) +
    guides(colour   = guide_legend(nrow=3,
                                   override.aes=list(linewidth=1.2)),
           linetype = guide_legend(nrow=3),
           shape    = guide_legend(nrow=1))
  ggsave(file.path(OUTPUT_DIR, sprintf("figB_per_dataset_cindex_%s.png", mech)),
         plot=pB, width=15, height=15, units="in", dpi=900, bg="white")
  ggsave(file.path(OUTPUT_DIR, sprintf("figB_per_dataset_cindex_%s.pdf", mech)),
         plot=pB, width=15, height=15, units="in", device=cairo_pdf)
  cat(sprintf("Saved: figB_per_dataset_cindex_%s (.png + .pdf)\n", mech))
}

# ── 20. Console Quick Checks ──────────────────────────────────────────────────
cat("\n=== METHOD-AVERAGED C-index & Uno's C @ 50% missingness ===\n")
cat("(ME_HAS: All classifiers; others: averaged across classifiers)\n")
grand_summ %>%
  dplyr::filter(abs(MissRate-0.50)<1e-6 | MissRate==0) %>%
  dplyr::group_by(MissMech, Method) %>%
  dplyr::summarise(Cindex_mean = mean(Cindex_mean, na.rm=TRUE),
                   Uno_C_mean  = mean(Uno_C_mean,  na.rm=TRUE),
                   .groups="drop") %>%
  dplyr::mutate(Method=factor(Method, levels=METHOD_ORDER)) %>%
  dplyr::arrange(MissMech, dplyr::desc(Cindex_mean)) %>%
  as.data.frame() %>% print(row.names=FALSE)

cat("\n=== PER-CLASSIFIER breakdown @ 50% missingness ===\n")
grand_summ %>%
  dplyr::filter(abs(MissRate-0.50)<1e-6 | MissRate==0) %>%
  dplyr::mutate(Method=factor(Method, levels=METHOD_ORDER)) %>%
  dplyr::arrange(MissMech, Method, Classifier) %>%
  dplyr::select(MissMech, Method, Classifier, Cindex_mean, Uno_C_mean) %>%
  as.data.frame() %>% print(row.names=FALSE)

# ── 21. Final Summary ─────────────────────────────────────────────────────────
cat("\n=============================================\n")
cat("  ALL DONE — ME-HAS Survival Benchmark\n")
cat("=============================================\n")
cat("Methods: Complete/MeanSI/KNN/RFI/MissInd/MIE_PMM_AVG/MIE_CART_AVG/MICE_SE/ME_HAS\n")
cat("Metrics: Harrell C-index (primary) | Uno's C via pec::cindex() (secondary)\n")
cat(sprintf("Bonferroni N = %d per mechanism (28 combos × 2 metrics × 5 rates)\n",
            N_BONF_PER_MECH))
cat(sprintf("\nOutput: %s/\n", OUTPUT_DIR))

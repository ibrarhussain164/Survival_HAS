# =============================================================================
# Mixed-Effects Hierarchical Aware Stacking (ME-HAS) — Survival Pipeline
# Script : Multi_HAS_Survival_Pipeline.R
# Version: 10 (v9 run; reporting: ME-HAS main = 4 MICE methods)
# Authors: Ibrar Hussain, Lichao Yang, Sohail Akhtar, Linkai Meil,
#          Shahzaib Rafiq, Can Liu, Yiqing Gu, Zaixiang Tang
# Affil. : Soochow University / University of Tabuk
# =============================================================================
#
# Pipeline overview:
#   - Nine benchmark right-censored survival datasets
#   - Two missing-data mechanisms: MAR, MCAR
#   - Five missingness proportions: 10%, 20%, 30%, 40%, 50%
#   - S = 10 stratified random train/test splits per condition
#   - MAIN (MAIN_HAS_VARIANT = "MM"):
#     MICE-SE (competitor, Aleryani et al. 2020): MICE-PMM, m = 5; per-learner
#       average over the 5 draws, free-weight meta-learner.
#     ME-HAS (proposed): 4 MICE methods (pmm, cart, norm, rf) x 5 draws, draws
#       averaged within learner x method (C=4 -> 16 streams);
#       coxme lp + (lp|clf_id) + (1|method).
#   - SUPPLEMENTARY: same-imputation comparison -- MICE-SE and ME-HAS both with
#     PMM only, and both with the 4 methods (reporting labels: MICE_SE, ME_HAS_PMM,
#     MICE_SE_MM, ME_HAS). Computation names: ME_HAS = PMM, ME_HAS_MM = 4-method.
#   - Methods: Complete (paired, same split) / MeanSI / KNN / RFI / MissInd /
#              MIE-PMM-Avg / MIE-CART-Avg / MICE-SE / ME-HAS
#   - Metrics: Harrell C-index, Brier score (tau), integrated Brier score
#   - Reporting: grand-mean tables + Kendall's W rank tables (irr::kendall)
#
# Design notes:
#   - Unified seed formula guarantees identical train/test partitions across
#     all methods within each experimental cell.
#   - Imputation-method random intercept (method_id BLUP) is excluded from
#     test-set prediction; only the fixed slope and classifier-specific
#     slope BLUP are applied (it is a per-method constant: no effect on ranking).
#   - Split-level .rds checkpoints allow resumption after interruptions.
#   - MICE (pmm/cart/norm/rf) uses the Nelson-Aalen cumulative hazard H0(T)
#     and the event indicator as auxiliary predictors (White & Royston, 2009).
#   - Missing Indicator: binary flag columns (_ind suffix) appended to
#     mean-imputed data; falls back to MeanSI if the model fails.
# =============================================================================

# ── 0. Packages ───────────────────────────────────────────────────────────────
# NOTE: pec aur flexsurv hata diye -- pipeline mein kahin use nahi hote the
# (Uno's C / pec::cindex ab report nahi hota; Weibull AFT survival::survreg se hai).
needed <- c("mice","VIM","missForest","survival","gbm",
            "coxme","caret","dplyr","tidyr","ggplot2","purrr",
            "readr","scales",
            "riskRegression","prodlim","ranger",  # ranger: mice "rf" method
            "irr",                                 # irr: Kendall's W (kendall())
            "rpart")                               # rpart: survival tree (weak learner)
missing_pkgs <- needed[!needed %in% installed.packages()[,"Package"]]
if (length(missing_pkgs) > 0) {
  message("Installing missing packages: ", paste(missing_pkgs, collapse=", "))
  # getOption("repos"): agar aap ne China mirror set kiya hai to wahi use hoga
  repos <- getOption("repos")
  if (is.null(repos) || identical(unname(repos["CRAN"]), "@CRAN@"))
    repos <- "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"
  install.packages(missing_pkgs, repos=repos)
}
suppressPackageStartupMessages({
  library(mice); library(VIM); library(missForest)
  library(survival); library(gbm)
  library(coxme); library(caret)
  library(dplyr); library(tidyr); library(ggplot2)
  library(purrr); library(readr); library(scales)
  library(riskRegression); library(prodlim); library(irr)
})
set.seed(2024)

# ── 1. Settings ───────────────────────────────────────────────────────────────
N_SPLITS   <- 10
MISS_RATES <- seq(0.10, 0.50, by=0.10)
M_IMP      <- 5
M_IMP_SE   <- 5
MAXIT_MICE <- 5           # mice default (5 iterations), binary MICE-GLMM-Stack pipeline jaisa
CV_FOLDS   <- 5
TRAIN_FRAC <- 0.70
# Base learners (4), har ek alag model family se:
#   CoxPH      -- semi-parametric (linear effects)
#   GBM        -- gradient-boosted trees (ensemble, Cox loss)
#   SurvTree   -- single survival tree (rpart)
#   WeibullAFT -- fully parametric (accelerated failure time)
# (RSF aur CoxSpline ka code fit_surv() mein mojood hai lekin library mein shamil nahi.)
LEARNER_SET <- "tree"   # output folder ke naam ke liye
SURV_NAMES  <- c("CoxPH", "GBM", "SurvTree", "WeibullAFT")
MISS_MECHS <- c("MAR", "MCAR")
METHOD_ORDER <- c("Complete","MeanSI","KNN","RFI","MissInd",
                  "MIE_PMM_AVG","MIE_CART_AVG","MICE_SE","ME_HAS")
# "All" classifier wale methods (sab classifiers ko combine karte hain)
ALL_CLF_METHODS <- c("ME_HAS")

# SUPPLEMENTARY: stacking methods ka multi-method (4 MICE methods) variant.
# Yeh main tables / figures / Kendall ranks mein NAHI aate -- sirf alag
# supplementary table mein (Multi_HAS_Survival_supp_multimethod_<mech>.tex/.csv).
RUN_MULTIMETHOD <- TRUE
SUPP_METHODS     <- c("MICE_SE_MM", "ME_HAS_MM")
METHOD_ORDER_ALL <- c(METHOD_ORDER, if (RUN_MULTIMETHOD) SUPP_METHODS)

# ── MAIN paper ka ME-HAS variant ──
# "MM"  = ME-HAS 4 MICE methods (pmm+cart+norm+rf) par -- MAIN paper (proposed method)
# "PMM" = ME-HAS sirf MICE-PMM draws par
# MICE_SE (competitor) MAIN mein hamesha Aleryani et al. (2020) jaisa sirf PMM hai.
# NOTE: sirf REPORTING ke labels badalte hain. Computation aur checkpoints ke andar
# naam wahi rehte hain (ME_HAS = PMM variant, ME_HAS_MM = 4-method variant), isliye
# naya run zaroori nahi -- checkpoints se tables dobara ban jaati hain.
# MAIN_HAS_VARIANT = "MM" par reporting labels:
#   "ME_HAS"     = 4-method variant (main tables, figures, Kendall's W)
#   "ME_HAS_PMM" = PMM-only variant (sirf supplementary table)
MAIN_HAS_VARIANT <- "MM"
if (MAIN_HAS_VARIANT == "MM" && !RUN_MULTIMETHOD)
  stop("MAIN_HAS_VARIANT = 'MM' ke liye RUN_MULTIMETHOD = TRUE hona chahiye")
REPORT_SUPP_METHODS <- if (MAIN_HAS_VARIANT == "MM") c("MICE_SE_MM", "ME_HAS_PMM") else SUPP_METHODS
REPORT_LEVELS       <- c(METHOD_ORDER, if (RUN_MULTIMETHOD) REPORT_SUPP_METHODS)

# ── 1b. ME-HAS meta-learner (coxme) ──────────────────────────────────────────
# Model: lp (fixed common slope) + classifier random slope + imputation random
# intercept:  Surv(time, event) ~ lp + (lp | clf_id) + (1 | method_id)
# (code mein variable ka naam method_id hai: main mein = PMM draw, supp mein = MICE method)
# coxme mein (lp | clf_id) SIRF random slope hai (intercept automatic add nahi hota).
#   pmm  = predictive mean matching      cart = classification/regression trees
#   norm = Bayesian linear regression    rf   = random-forest imputation
# Sab MICE methods mein White & Royston (2009) auxiliaries: Nelson-Aalen H0(T) + event.
# COMPUTATION mein dono variants chalte hain (neeche). Kaunsa variant MAIN paper
# mein ME-HAS kehlata hai, yeh MAIN_HAS_VARIANT (section 1) decide karta hai.
# Computation ka PMM variant (Aleryani et al. 2020 jaisa): ME-HAS aur MICE_SE dono
#   MICE-PMM (m = 5) use karte hain.
#   ME-HAS : streams = C learners x 5 PMM draws (C = 4 -> 20);
#            coxme  lp + (lp | clf_id) + (1 | draw)
#   MICE_SE: L0 = har classifier ki 5 PMM draws ki OOF/test predictions ka average
# SUPPLEMENTARY (RUN_MULTIMETHOD = TRUE): ME_HAS_MM / MICE_SE_MM
#   4 MICE methods (pmm, cart, norm, rf) x 5 draws; draws ka average per
#   learner x method -> C x 4 streams (C = 4 -> 16); ME-HAS_MM: lp + (lp|clf_id) + (1|method)
#   Dono stacking methods ke inputs (har variant ke andar) bilkul same hain.
MICE_STACK_METHODS <- c("pmm", "cart", "norm", "rf")   # 4-method variant (MAIN ME-HAS)

# Boundary (singular) fit ki definition.
# coxme mein isSingular() nahi hai; coxme variance ko log scale par optimise
# karta hai (theta = log(var), exp(pmax(-36, ...))), isliye variance kabhi
# exactly 0 nahi hoti -- boundary par woh bohat chhota positive number deti hai.
#   SINGULAR  : SD < 1e-4  -- bilkul lme4::isSingular(tol = 1e-4) jaisa (lme4
#               yeh tolerance SD/theta par lagata hai) -> binary pipeline se consistent
#   NEAR      : SD < 0.01  -- descriptive "near-boundary" (practically zero)
SD_SINGULAR_TOL <- 1e-4
SD_NEAR_TOL     <- 1e-2

# ── 1c. Calibration / Brier settings ─────────────────────────────────────────
# Har method sirf risk score deta hai (scales mukhtalif). Brier/calibration ke
# liye S(t|x) chahiye, isliye TEST set par K-fold cross-fitted recalibration:
#   har fold ke liye baaki (K-1) folds par coxph(Surv ~ score) + Breslow fit,
#   phir is fold ke subjects ka S(t|x) predict (riskRegression::predictRisk).
# Sab methods ke liye BILKUL same folds aur same tareeqa -- scale-free aur fair.
CALIB_FOLDS   <- 5
CALIB_SEED    <- 4242        # fold assignment ka seed (global RNG ko disturb nahi karta)
CALIB_TAU_Q   <- 0.75        # tau = test event times ka 75th percentile
# IBS grid: riskRegression IBS ko sirf diye gaye `times` par step function ki
# tarah integrate karta hai (computePerformance.R: Dint(x=c(0,times), y=c(0,Brier)))
# aur pehle interval [0, t1) ki value 0 leta hai. Isliye grid GHANA hona chahiye
# aur t1 chhota -- warna IBS neeche ki taraf biased. 100 event-time quantiles (0..tau].
CALIB_GRID_N  <- 100
CALIB_METRICS <- c("Brier","IBS")

# ── Evaluation metrics (sirf yeh teen) ──
#   Cindex : Harrell C-index (survival::concordance)        -- zyada behtar
#   Brier  : Brier score at tau (riskRegression::Score)     -- kam behtar
#   IBS    : Integrated Brier score 0..tau (riskRegression) -- kam behtar
EVAL_METRICS <- c("Cindex", "Brier", "IBS")
METRIC_HIB   <- c(Cindex=TRUE, Brier=FALSE, IBS=FALSE)   # higher-is-better?
METRIC_LABEL <- c(Cindex="Harrell C-index", Brier="Brier score", IBS="Integrated Brier score")

# Naya folder: purane "v9" checkpoints mein ME-HAS ke results prediction-bug
# wale hain (classifier slope ko intercept ki tarah joda gaya tha), isliye woh
# dobara use NAHI hone chahiye.
# v9_multimeth: ME-HAS ab 4 imputation methods (pmm/cart/norm/rf) use karta hai.
# Naya folder, taa ke pichhle (sirf-PMM) run ke checkpoints load na hon.
# v9_final: split-seed fix ke baad naya folder (purane checkpoints alag splits ke hain)
RUN_ID         <- paste0("v9_final_", LEARNER_SET)
OUTPUT_DIR     <- paste0("surv_HAS_", RUN_ID)
CHECKPOINT_DIR <- file.path(OUTPUT_DIR, "checkpoints")
DATA_DIR       <- file.path(path.expand("~"), "survival_data")
DATA_DIR2      <- file.path(getwd(), "survival_data", "data")
dir.create(OUTPUT_DIR,     recursive=TRUE, showWarnings=FALSE)
dir.create(CHECKPOINT_DIR, recursive=TRUE, showWarnings=FALSE)
# Har R session ki apni diagnostics file (parallel sessions ek doosre ki file
# overwrite na karein). Summary saari "_part_*.csv" files ko jor kar banti hai.
SESSION_TAG    <- paste0(format(Sys.time(), "%Y%m%d%H%M%S"), "_", Sys.getpid())
META_DIAG_FILE <- file.path(OUTPUT_DIR, sprintf("Multi_HAS_Survival_mehas_diagnostics_part_%s.csv", SESSION_TAG))
RESET_CHECKPOINTS <- FALSE
if (RESET_CHECKPOINTS && dir.exists(CHECKPOINT_DIR)) {
  old_ckpts <- list.files(CHECKPOINT_DIR, pattern="\\.rds$", full.names=TRUE)
  if (length(old_ckpts) > 0) {
    file.remove(old_ckpts)
    cat(sprintf(">>> RESET: deleted %d checkpoint(s)\n", length(old_ckpts)))
  }
}

# ── 2. Seed functions ─────────────────────────────────────────────────────────
# FIX: pehle train/test split ka seed miss_rate aur mech par bhi depend karta
# tha, isliye har rate ka split ALAG tha aur 10% -> 50% trajectories mein split
# ka noise mila hua tha (Complete ka C-index rate ke saath badalta tha).
# Ab teen alag seeds:
#   partition_seed: sirf (split, dataset) -- har rate aur mechanism par SAME
#                   train/test split, taa ke rates ke beech comparison paired ho
#   complete_seed : sirf (split, dataset) -- Complete ke model fits har rate par
#                   bilkul same (GBM ki randomness bhi same)
#   cell_seed     : (split, dataset, rate, mech) -- missingness mask aur MICE
#                   imputations ke liye (har cell ka apna mask)
ds_seed_offset <- function(ds_name) sum(utf8ToInt(ds_name))   # dataset-specific
make_partition_seed <- function(sp, ds_name) 100000L + sp * 137L + ds_seed_offset(ds_name)
make_complete_seed  <- function(sp, ds_name) 200000L + sp * 137L + ds_seed_offset(ds_name)
make_split_seed <- function(sp, ds_name, miss_rate=0, mech="NONE") {
  mech_offset <- switch(mech, MAR=9999L, MCAR=19999L, 0L)
  sp * 137L + as.integer(round(miss_rate * 1000)) + ds_seed_offset(ds_name) + mech_offset
}

# ── 3. Dataset Loaders ────────────────────────────────────────────────────────
make_surv_ds <- function(X_num, time, event) {
  df <- data.frame(X_num)
  df$time  <- as.numeric(time)
  df$event <- as.integer(event)
  df <- df[complete.cases(df[, c("time","event")]) & df$time > 0, ]
  pred_cols <- setdiff(names(df), c("time","event"))
  df <- df[, c(pred_cols[sapply(df[, pred_cols, drop=FALSE], is.numeric)], "time","event")]
  # FIX: kuch datasets (misal PBC) mein covariates mein PEHLE SE missing values
  # hain. Benchmark ka design yeh hai ke missingness hum khud (known mechanism
  # aur rate se) induce karein, isliye shuru complete data se hona chahiye.
  # Warna: (1) actual missingness nominal se zyada (PBC: 57.6% vs 50%),
  # (2) test set mein bhi NA -> un subjects ki predictions NA, aur
  # (3) "Complete" method asal mein complete nahi hota.
  n0 <- nrow(df)
  df <- df[stats::complete.cases(df), , drop=FALSE]
  if (nrow(df) < n0)
    message(sprintf("  [DATA] %d rows with pre-existing missing covariates removed (complete cases: %d of %d)",
                    n0 - nrow(df), nrow(df), n0))
  df
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

# ── Parallel run: datasets ko kai R sessions mein baanto ──
# RUN_DATASETS <- NULL                       -> saare datasets (ya final "combine" run)
# RUN_DATASETS <- c("WHAS500", "GBSG2")      -> sirf yeh datasets (ek session)
# Har session same OUTPUT_DIR mein apne checkpoints likhta hai. Aakhir mein
# RUN_DATASETS <- NULL ke saath ek dafa chalayein: saare datasets checkpoints se
# load ho kar final tables / figures / Kendall ban jate hain (sirf kuch minute).
RUN_DATASETS <- NULL
# Terminal se bhi de sakte hain:  Rscript Multi_HAS_Survival_Pipeline.R WHAS500,GBSG2,PBC
.cli_args <- commandArgs(trailingOnly = TRUE)
if (length(.cli_args) > 0 && nzchar(.cli_args[1]))
  RUN_DATASETS <- trimws(strsplit(.cli_args[1], ",")[[1]])
if (!is.null(RUN_DATASETS)) {
  bad <- setdiff(RUN_DATASETS, sapply(ALL_DATASETS, `[[`, "name"))
  if (length(bad)) stop("Ghalat dataset naam: ", paste(bad, collapse=", "))
  DATASET_REGISTRY <- Filter(function(d) d$name %in% RUN_DATASETS, ALL_DATASETS)
  cat(sprintf(">>> Is session mein datasets: %s\n",
              paste(sapply(DATASET_REGISTRY, `[[`, "name"), collapse=", ")))
}

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

# Nelson-Aalen cumulative hazard H0(T_i) har subject ke apne time par
# (White & Royston 2009 ka recommended MICE auxiliary).
# FIX: purana code survfit(type="fh") + -log(surv) use karta tha. type="fh"
# partial-match se "fh2" (Efron tie-corrected) ban jata hai, aur -log(S) asal
# Nelson-Aalen nahi. survfit() ka $cumhaz hamesha Nelson-Aalen (ctype=1) hai.
nelson_aalen_at <- function(time_vec, event_vec) {
  n <- length(time_vec)
  tryCatch({
    fit <- survival::survfit(survival::Surv(time_vec, event_vec) ~ 1)
    H   <- stats::stepfun(fit$time, c(0, fit$cumhaz))
    out <- H(time_vec); out[!is.finite(out)] <- 0; out
  }, error=function(e) rep(0, n))
}

# MICE ke baad koi NA bach jaye (norm/rf mein collinearity wagaira) to
# training mean se bharo -- binary pipeline ka fix_residual_nas() jaisa.
fix_residual_nas <- function(df) {
  for (nm in names(df)) if (any(is.na(df[[nm]]))) {
    fv <- mean(df[[nm]], na.rm=TRUE); if (!is.finite(fv)) fv <- 0
    df[[nm]][is.na(df[[nm]])] <- fv
  }
  df
}

# Ek hi MICE method (pmm/cart/norm/rf) se M imputations, survival auxiliaries ke saath
mice_impute_surv_method <- function(X_miss, time_vec, event_vec, method,
                                    m=M_IMP, seed=1, maxit=MAXIT_MICE) {
  na_H  <- nelson_aalen_at(time_vec, event_vec)
  df    <- data.frame(X_miss, .na_H=na_H, .event=event_vec)
  meths <- setNames(rep(method, ncol(df)), names(df))
  meths[c(".na_H",".event")] <- ""                 # auxiliaries complete hain
  for (nm in names(df)) if (!any(is.na(df[[nm]]))) meths[nm] <- ""
  pm <- mice::make.predictorMatrix(df)
  pm[c(".na_H",".event"), ] <- 0L                  # auxiliaries impute nahi hote
  suppressMessages(suppressWarnings(
    mids <- mice::mice(df, m=m, method=meths, predictorMatrix=pm,
                       seed=seed, printFlag=FALSE, maxit=maxit)
  ))
  X_cols <- names(X_miss)
  lapply(seq_len(m), function(i)
    fix_residual_nas(mice::complete(mids, i)[, X_cols, drop=FALSE]))
}

# Purane naam (MIE_PMM_AVG / MICE_SE ke liye) -- ab general function se
mice_impute_surv <- function(X_miss, time_vec, event_vec,
                             m=M_IMP, seed=1, maxit=MAXIT_MICE)
  mice_impute_surv_method(X_miss, time_vec, event_vec, "pmm", m, seed, maxit)

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
  tryCatch(mice_impute_surv_method(X_miss, time_vec, event_vec, "cart", m, seed, maxit),
           error=function(e) {
             message("  [CART fallback → PMM]")
             mice_impute_surv(X_miss, time_vec, event_vec, m, seed, maxit)
           })
}

# ── 6. Survival Model Wrapper ─────────────────────────────────────────────────
# ── Learner failure log ──
# Pehle learner errors chupke se NA ban jate the. Ab har error count hota hai,
# har learner ke pehle 3 errors console mein print hote hain, aur end par
# Multi_HAS_Survival_learner_failures.csv save hoti hai.
.FIT_LOG <- new.env()
.FIT_LOG$n   <- list()   # learner -> error count
.FIT_LOG$fb  <- list()   # learner -> fallback count (CoxSpline -> linear)
.FIT_LOG$msg <- list()   # learner -> pehle kuch error messages
log_fit_event <- function(kind, learner, msg) {
  cur <- .FIT_LOG[[kind]][[learner]]
  .FIT_LOG[[kind]][[learner]] <- (if (is.null(cur)) 0 else cur) + 1
  if (kind == "n") {
    k <- .FIT_LOG$n[[learner]]
    if (k <= 3) message(sprintf("  [LEARNER ERR] %s (#%d): %s", learner, k, msg))
    if (length(.FIT_LOG$msg[[learner]]) < 10)
      .FIT_LOG$msg[[learner]] <- c(.FIT_LOG$msg[[learner]], msg)
  }
}

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
    } else if (model_name == "CoxSpline") {
      # Additive Cox model: har continuous predictor (>= 10 unique values) par
      # penalised spline (survival::pspline, df = 3); baaki predictors linear.
      # Non-linear effects pakadta hai jo CoxPH (linear) nahi pakadta.
      terms <- vapply(p_names, function(v)
        if (length(unique(df_tr[[v]])) >= 10) sprintf("pspline(%s, df = 3)", v) else v,
        character(1))
      f_sp <- as.formula(paste("Surv(time, event) ~", paste(terms, collapse = " + ")))
      lp <- tryCatch({
        m <- survival::coxph(f_sp, data=df_tr, ties="efron",
                             control=coxph.control(iter.max=150, toler.inf=0.75))
        out <- as.numeric(predict(m, newdata=df_te, type="lp"))
        if (any(!is.finite(out))) stop("non-finite spline predictions")
        out
      }, error=function(e) NULL)
      if (is.null(lp)) {
        # Penalised spline fit fail -> linear CoxPH (fallback count hota hai,
        # end par learner_failures CSV mein report)
        log_fit_event("fb", "CoxSpline", "spline fit fail -> linear")
        m  <- survival::coxph(f_surv, data=df_tr, ties="efron",
                              control=coxph.control(iter.max=150, toler.inf=0.75))
        lp <- predict(m, newdata=df_te, type="lp")
      }
      lp
    } else if (model_name == "RSF") {
      # Random survival forest (ranger, logrank split). Risk score = ensemble
      # "mortality" = har subject ki cumulative hazard ka sum over unique death
      # times (Ishwaran et al. 2008). Positive aur skewed hai -> log lete hain,
      # taa ke baaki learners ki tarah log-hazard jaise scale par ho.
      m <- ranger::ranger(f_surv, data=df_tr, num.trees=200, min.node.size=15,
                          splitrule="logrank", num.threads=1, verbose=FALSE,
                          seed=sample.int(.Machine$integer.max, 1))
      chf <- predict(m, data=df_te)$chf
      log(pmax(rowSums(chf), 1e-8))
    } else if (model_name == "SurvTree") {
      # Single survival tree (rpart, method="exp"; Therneau & Atkinson), standard settings.
      # predict() estimated relative EVENT RATE deta hai (positive) -> log lete
      # hain taa ke baaki learners ki tarah log-hazard scale par ho.
      m <- rpart::rpart(f_surv, data=df_tr, method="exp",
                        control=rpart::rpart.control(cp=0.01, minbucket=10, xval=0))
      log(pmax(as.numeric(predict(m, newdata=df_te, type="vector")), 1e-8))
    } else if (model_name == "WeibullAFT") {
      m <- survival::survreg(f_surv, data=df_tr, dist="weibull")
      -predict(m, newdata=df_te, type="lp")
    } else rep(NA_real_, nrow(df_te))
  }, error=function(e) {
    log_fit_event("n", model_name, conditionMessage(e))
    rep(NA_real_, nrow(df_te))
  }))
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
  
  # Harrell's C: survival::concordance (Therneau). reverse=TRUE ka matlab:
  # zyada risk score = kam survival time (concordant). Ties: survival ka default.
  cindex <- tryCatch(
    survival::concordance(
      survival::Surv(time_te, event_te) ~ risk_scores, reverse=TRUE)$concordance,
    error=function(e) NA_real_)

  calib <- tryCatch(compute_calib_metrics(risk_scores, time_te, event_te),
                    error=function(e) {
                      message(sprintf("  [CALIB WARN] %s", conditionMessage(e)))
                      setNames(rep(NA_real_, length(CALIB_METRICS)), CALIB_METRICS)
                    })
  c(Cindex=as.numeric(cindex), calib)
}

# ── 7b. Calibration & overall-accuracy metrics (library-based) ───────────────
# Sab metrics established packages se:
#   Brier(tau), IBS(0..tau) -> riskRegression::Score (IPCW, Kaplan-Meier censoring)
#                              Graf et al. (1999); Gerds & Schumacher (2006)
#
# IMPORTANT (paper mein likhna hai): yeh RECALIBRATED performance hai -- score se
# probability ka mapping test folds par cross-fit hota hai (sab methods ke liye
# same folds). Brier/IBS overall accuracy (discrimination + calibration) naapte
# hain, recalibration ke baad.

# Local seed: fold assignment reproducible, lekin global RNG stream wahi rehti hai
# (warna baaki methods ke results badal jate).
with_local_seed <- function(seed, expr) {
  had  <- exists(".Random.seed", envir=globalenv(), inherits=FALSE)
  old  <- if (had) get(".Random.seed", envir=globalenv()) else NULL
  on.exit({
    if (had) assign(".Random.seed", old, envir=globalenv())
    else if (exists(".Random.seed", envir=globalenv(), inherits=FALSE))
      rm(".Random.seed", envir=globalenv())
  })
  set.seed(seed)
  expr
}

# Test set ke folds -- sirf test data (time/event) par depend, method par NAHI,
# isliye har method ko bilkul same folds milte hain.
calib_folds <- function(time_te, event_te) {
  seed <- CALIB_SEED + length(time_te) + sum(event_te) +
    as.integer(round(sum(time_te) %% 1e5))
  with_local_seed(seed,
    caret::createFolds(factor(event_te), k=CALIB_FOLDS, list=TRUE, returnTrain=FALSE))
}

# Cross-fitted S(t|x): har fold ke liye baaki folds par coxph(score) + Breslow
crossfit_event_prob <- function(risk, time_te, event_te, times) {
  n    <- length(risk)
  risk <- as.numeric(risk)
  pmat <- matrix(NA_real_, n, length(times))
  for (fi in calib_folds(time_te, event_te)) {
    tri <- setdiff(seq_len(n), fi)
    mu  <- mean(risk[tri], na.rm=TRUE); s <- sd(risk[tri], na.rm=TRUE)
    if (!is.finite(s) || s < 1e-12) s <- 1
    z   <- (risk - mu) / s
    z[!is.finite(z)] <- 0              # NA score = average risk (z = 0)
    d_tr <- data.frame(time=time_te[tri], event=event_te[tri], z=z[tri])
    if (stats::sd(d_tr$z) < 1e-12) {
      # Score constant hai (misal SurvTree ne koi split nahi banaya): score mein
      # koi information nahi -> sab subjects ko Kaplan-Meier (null model) event
      # probability -- "no prognostic information" ka sahi nateeja.
      km <- survival::survfit(Surv(time, event) ~ 1, data=d_tr)
      S  <- summary(km, times=times, extend=TRUE)$surv
      pmat[fi, ] <- matrix(1 - S, nrow=length(fi), ncol=length(times), byrow=TRUE)
      next
    }
    fit  <- survival::coxph(Surv(time, event) ~ z, data=d_tr, x=TRUE, y=TRUE, ties="breslow")  # riskRegression docs: x=TRUE, y=TRUE
    pmat[fi, ] <- riskRegression::predictRisk(fit, newdata=data.frame(z=z[fi]),
                                              times=times)
  }
  pmat   # event probability F(t|x) = 1 - S(t|x)
}

compute_calib_metrics <- function(risk_scores, time_te, event_te) {
  ev  <- time_te[event_te == 1L & is.finite(time_te)]
  if (length(ev) < 10L) stop("calibration ke liye kam az kam 10 events chahiye")
  tau  <- as.numeric(stats::quantile(ev, CALIB_TAU_Q))
  # Ghana grid: event times ke quantiles 0 se tau tak (tau khud aakhri point)
  grid <- unique(as.numeric(stats::quantile(ev[ev <= tau],
                                            seq(0, 1, length.out=CALIB_GRID_N))))
  grid <- sort(unique(c(grid[grid > 0 & grid < tau], tau)))
  stopifnot(abs(grid[length(grid)] - tau) < 1e-12)   # tau hamesha aakhri point

  pmat <- crossfit_event_prob(risk_scores, time_te, event_te, grid)
  df   <- data.frame(time=time_te, event=as.integer(event_te))

  # Brier(tau) aur IBS(0..tau) -- riskRegression::Score (IPCW, KM censoring),
  # cross-fitted probabilities par
  sc <- riskRegression::Score(
    list(Model=pmat), formula=Surv(time, event) ~ 1, data=df, times=grid,
    metrics="brier", summary="ibs",
    cens.model="km", null.model=FALSE, se.fit=FALSE, progress.bar=NULL)
  br  <- as.data.frame(sc$Brier$score)
  br  <- br[br$model == "Model" & abs(br$times - tau) < 1e-10, ]

  c(Brier = if (nrow(br)) br$Brier[1] else NA_real_,
    IBS   = if (nrow(br) && "IBS" %in% names(br)) br$IBS[1] else NA_real_)
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
# Streams: har classifier c x imputation method k (pmm/cart/norm/rf) ka ek
# stream = us method ke M draws ki predictions ka average.
#
# Imputation-method random intercept v_k test prediction mein shamil NAHI:
# test data complete hai (X_te_full); "method_id=k" sirf yeh batata hai ke
# model kis training imputation method par train hua -- test subject ki koi
# property nahi. Mathematical note: v_k har method ke liye ek constant hai;
# har test subject ke paas sab K methods ki rows hain, isliye average ke baad
# v_k ka hissa sab subjects ke liye same constant -- ranking (C-index)
# par asar exactly zero.
# Final score: mean over (c, k) of (beta + u_c) * lp_ick.

# ── 10a. Long-format data  ─────────────────────
# Har subject i, classifier c, draw m ke liye ek row: n x C x M rows.
# lp har classifier ke andar training moments se standardise hota hai,
# kyunke classifiers ke scales mukhtalif hain (log-HR, GBM score, log event rate,
# -log time). Bina standardisation ke common slope beta ka koi matlab nahi.
# oof_list[[clf]] / test_list[[clf]]: matrix, har column ek imputation METHOD
# ka stream (us method ke M draws ka average). Column names = method names.
build_has_long <- function(oof_list, test_list, time_tr, event_tr) {
  clfs <- names(oof_list)
  m    <- ncol(oof_list[[1]])
  mlev <- if (!is.null(colnames(oof_list[[1]]))) colnames(oof_list[[1]]) else as.character(seq_len(m))
  n_tr <- length(time_tr)
  n_te <- nrow(test_list[[1]])

  long_tr <- do.call(rbind, lapply(clfs, function(cl)
    do.call(rbind, lapply(seq_len(m), function(k)
      data.frame(lp     = oof_list[[cl]][, k],
                 time   = time_tr,
                 event  = event_tr,
                 obs_id = factor(seq_len(n_tr)),
                 method_id = factor(mlev[k], levels=mlev),
                 clf_id = factor(cl, levels=clfs))))))

  # Per-classifier normalisation (sirf training moments)
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


  long_te <- do.call(rbind, lapply(clfs, function(cl)
    do.call(rbind, lapply(seq_len(m), function(k)
      data.frame(
        lp      = (test_list[[cl]][, k] - clf_norm[[cl]]$mu) / clf_norm[[cl]]$s,
        obs_id  = factor(seq_len(n_te)),
        method_id = factor(mlev[k], levels=mlev),
        clf_id  = factor(cl, levels=clfs),
        obs_int = seq_len(n_te))))))
  long_te$lp[!is.finite(long_te$lp)] <- 0

  list(long_tr=long_tr, long_te=long_te, clfs=clfs)
}

# ── 10b. Test-set prediction  ──────────────────
# BUG FIX (purane code mein): classifier ka random effect u_c ek SLOPE hai
# (lp | clf_id), lekin purana code usay  beta*lp + u_c  ki tarah intercept bana
# kar jorta tha. Har test subject ke paas sab C x M rows hoti hain, isliye
# average ke baad u_c ka hissa har subject ke liye ek hi constant ban jata tha,
# aur ranking (C-index) par uska koi asar nahi tha. Natija: ME-HAS asal mein
# sirf standardised lp ka simple average ban jata tha.
# Sahi formula: (beta + u_c) * lp  -- yani har classifier ka apna weight.
# Draw intercept v_m test prediction mein shamil nahi (paper Section 2.6).
predict_has_long <- function(long_te, beta_lp, slope_clf) {
  w_c <- beta_lp + slope_clf[as.character(long_te$clf_id)]
  long_te$pred <- w_c * long_te$lp
  long_te$pred[!is.finite(long_te$pred)] <- 0
  agg <- aggregate(pred ~ obs_int, long_te, mean)
  agg[order(agg$obs_int), "pred"]
}

# Fallback: model fail ho to standardised lp ka simple average
fallback_has_pred <- function(long_te) {
  agg <- aggregate(lp ~ obs_int, long_te, mean)
  agg[order(agg$obs_int), "lp"]
}

EMPTY_VC <- list(var_slope_clf=NA_real_, var_intercept_imp=NA_real_)

# ── 10c. ME-HAS diagnostics log ──────────────────────────────────────────────
# Har fit ke foran baad CSV mein append hota hai (crash / resume ke baad bhi
# poora record): kaunsa formula fit hua, beta, classifier weights, variance
# components, aur kya variance boundary (singular) par thi.
.META_ENV <- new.env()
.META_ENV$cell    <- NA_character_
.META_ENV$variant <- "PMM"
log_meta_diag <- function(row) {
  old <- if (file.exists(META_DIAG_FILE))
    tryCatch(read.csv(META_DIAG_FILE, stringsAsFactors=FALSE), error=function(e) NULL)
  new <- if (is.null(old)) row else dplyr::bind_rows(old, row)
  write.csv(new, META_DIAG_FILE, row.names=FALSE)
}

# ── 10d. ME-HAS with coxme (bug fix ke saath) ─────────────────
# coxme syntax: (lp | clf_id) SIRF random slope hai -- coxme mein intercept
# automatic add nahi hota (coxme vignette). Fallback mein (1|clf_id) hata diya
# kyunke standardisation ke baad woh by construction ~0 hota hai.
run_has_surv_coxme <- function(oof_list, test_list, time_tr, event_tr) {
  L <- build_has_long(oof_list, test_list, time_tr, event_tr)
  long_tr <- L$long_tr; clfs <- L$clfs

  ok <- FALSE; mod <- NULL; frm_used <- NULL
  for (frm in list(
    Surv(time,event) ~ lp + (lp | clf_id) + (1 | method_id),
    Surv(time,event) ~ lp + (lp | clf_id),
    Surv(time,event) ~ lp + (1  | method_id)
  )) {
    if (ok) break
    suppressMessages(suppressWarnings(tryCatch({
      mod <- coxme::coxme(frm, data=long_tr); ok <- TRUE; frm_used <- frm
    }, error=function(e) NULL)))
  }
  if (!ok) {
    log_meta_diag(data.frame(Cell=.META_ENV$cell, Variant=.META_ENV$variant,
                             n_rows=nrow(long_tr),
                             formula="NONE (fallback simple mean)",
                             stringsAsFactors=FALSE))
    return(list(pred=fallback_has_pred(L$long_te), varcomp=EMPTY_VC,
                fallback=TRUE, engine_used="fallback_mean"))
  }

  beta_lp <- tryCatch({
    fe <- coxme::fixef(mod)
    v  <- if ("lp" %in% names(fe)) fe["lp"] else fe[1]
    if (is.finite(v)) as.numeric(v) else 1
  }, error=function(e) 1)

  re <- tryCatch(coxme::ranef(mod), error=function(e) list())
  slope_clf <- setNames(rep(0, length(clfs)), clfs)
  if ("clf_id" %in% names(re)) {
    rv <- re[["clf_id"]]
    if (is.data.frame(rv) || is.matrix(rv)) {
      nm <- rownames(rv); rv <- rv[, 1]; names(rv) <- nm
    } else rv <- unlist(rv)
    common <- intersect(names(rv), clfs)
    slope_clf[common] <- rv[common]
  }

  vc      <- extract_coxme_varcomp(mod)
  frm_txt <- paste(deparse(frm_used), collapse="")
  # Kaunsa formula fit hua -- full model ke ilawa koi bhi formula reduced model
  # hai aur paper mein report hona chahiye.
  log_meta_diag(data.frame(
    Cell = .META_ENV$cell, Variant = .META_ENV$variant, n_rows = nrow(long_tr),
    formula = frm_txt,
    full_model = grepl("clf_id", frm_txt) && grepl("method_id", frm_txt),
    n_methods = nlevels(long_tr$method_id),
    methods = paste(levels(long_tr$method_id), collapse="+"),
    beta = beta_lp,
    t(setNames(beta_lp + slope_clf, paste0("w_", names(slope_clf)))),
    var_slope_clf = vc$var_slope_clf, var_intercept_method = vc$var_intercept_imp,
    sd_slope_clf        = sqrt(vc$var_slope_clf),
    sd_intercept_method = sqrt(vc$var_intercept_imp),
    singular_clf    = isTRUE(sqrt(vc$var_slope_clf)     < SD_SINGULAR_TOL),
    singular_method = isTRUE(sqrt(vc$var_intercept_imp) < SD_SINGULAR_TOL),
    near_clf        = isTRUE(sqrt(vc$var_slope_clf)     < SD_NEAR_TOL),
    near_method     = isTRUE(sqrt(vc$var_intercept_imp) < SD_NEAR_TOL),
    stringsAsFactors = FALSE, row.names = NULL))

  list(pred    = predict_has_long(L$long_te, beta_lp, slope_clf),
       varcomp = vc, fallback = FALSE, engine_used = frm_txt)
}

# ── 10e. ME-HAS entry point ──────────────────────────────────────────────────
run_has_surv <- function(oof_list, test_list, time_tr, event_tr) {
  run_has_surv_coxme(oof_list, test_list, time_tr, event_tr)
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
  # var_intercept_imp column naam purana rakha (tables/CSV ke liye), lekin
  # ab yeh imputation METHOD (pmm/cart/norm/rf) ke random intercept ki variance hai
  list(var_slope_clf    = get_var("clf_id"),
       var_intercept_imp = get_var("method_id"))
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
  # FAIRNESS FIX: ME-HAS ki tarah har column (classifier) ko TRAINING moments
  # se standardise karo (test par wahi mu/s). Classifiers ke scales mukhtalif
  # hain; bina standardisation ke coxph meta-learner ki convergence kharab hoti
  # hai aur woh zyada fallback par jata hai. Fallback (rowMeans) raw scale par hi.
  L0_tr_raw <- L0_tr; L0_te_raw <- L0_te
  mu <- colMeans(L0_tr, na.rm=TRUE)
  sg <- apply(L0_tr, 2, sd, na.rm=TRUE)
  sg[!is.finite(sg) | sg < 1e-8] <- 1
  L0_tr <- sweep(sweep(as.matrix(L0_tr), 2, mu, "-"), 2, sg, "/")
  L0_te <- sweep(sweep(as.matrix(L0_te), 2, mu, "-"), 2, sg, "/")
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
  }, error=function(e) rowMeans(L0_te_raw, na.rm=TRUE))
}

# ── 12. Single Run ────────────────────────────────────────────────────────────
run_one_surv <- function(miss_mech, miss_rate, split_seed,
                         X_tr_full, time_tr, event_tr,
                         X_te_full, time_te, event_te,
                         X_tr_miss = NULL, complete_seed = NULL) {
  rows <- list()
  
  rec <- function(method, surv_name, pred) {
    mt <- compute_surv_metrics(pred, time_te, event_te)
    data.frame(Method=method, Classifier=surv_name, t(mt), stringsAsFactors=FALSE)
  }
  
  # ── Complete (PAIRED reference): bilkul isi train/test split par, poora data ──
  # Pehle Complete alag split (miss_rate=0 seed) par chalta tha, isliye missing
  # cells ke saath paired comparison nahi tha. Ab har cell mein same split.
  # complete_seed: har rate par same Complete fits (GBM ki randomness bhi).
  # Uske baad cell_seed se RNG dobara set -- baaki methods cell-specific rehte hain.
  if (!is.null(complete_seed)) set.seed(complete_seed)
  for (sn in SURV_NAMES)
    rows[[length(rows)+1]] <- rec("Complete", sn,
      safe_fit_surv(sn, X_tr_full, time_tr, event_tr, X_te_full))
  set.seed(split_seed + 7L)

  # Missingness mask main loop mein EK dafa banta hai aur yahan pass hota hai,
  # taa ke log ki gayi actual missingness wahi ho jis par sab methods chalte hain.
  if (is.null(X_tr_miss))
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
  
  # ════════════════════════════════════════════════════════════════════════
  # STACKING METHODS (MICE_SE aur ME-HAS)
  # ════════════════════════════════════════════════════════════════════════
  # Helper: invalid streams (>10% non-finite, kisi bhi classifier mein) hatao,
  # taa ke har classifier ke paas same streams hon aur koi stream chupke se 0 na bane.
  clean_streams <- function(so, st, tag) {
    ok <- Reduce(intersect, lapply(SURV_NAMES, function(sn)
      colnames(so[[sn]])[colMeans(is.finite(so[[sn]])) >= 0.9 &
                         colMeans(is.finite(st[[sn]])) >= 0.9]))
    if (length(ok) == 0) {
      fr <- sapply(SURV_NAMES, function(sn)
        round(pmin(colMeans(is.finite(so[[sn]])), colMeans(is.finite(st[[sn]]))), 2))
      stop(sprintf("[%s] koi stream valid nahi; finite fraction: %s", tag,
                   paste(capture.output(print(fr)), collapse=" | ")))
    }
    if (length(ok) < ncol(so[[1]]))
      message(sprintf("  [STACK %s] streams hataye: %s", tag,
                      paste(setdiff(colnames(so[[1]]), ok), collapse=",")))
    list(oof = lapply(so, function(m) m[, ok, drop=FALSE]),
         te  = lapply(st, function(m) m[, ok, drop=FALSE]))
  }

  # Helper: MICE_SE (Aleryani et al. 2020): har classifier ka L0 = us ke saare
  # streams (imputations) ka average; phir free-weight meta-learner (har learner).
  run_se_variant <- function(S, label) {
    l0 <- tryCatch({
      L0_tr <- sapply(SURV_NAMES, function(sn) rowMeans(S$oof[[sn]], na.rm=TRUE))
      L0_te <- sapply(SURV_NAMES, function(sn) rowMeans(S$te[[sn]],  na.rm=TRUE))
      L0_tr <- matrix(L0_tr, ncol=length(SURV_NAMES), dimnames=list(NULL, SURV_NAMES))
      L0_te <- matrix(L0_te, ncol=length(SURV_NAMES), dimnames=list(NULL, SURV_NAMES))
      L0_tr[!is.finite(L0_tr)] <- 0; L0_te[!is.finite(L0_te)] <- 0
      list(L0_tr=L0_tr, L0_te=L0_te)
    }, error=function(e) {
      message(sprintf("  [FALLBACK] %s L0 fail: %s", label, conditionMessage(e)))
      NULL
    })
    for (meta_sn in SURV_NAMES) {
      risk_se <- if (!is.null(l0)) {
        tryCatch(run_mice_se_surv(l0$L0_tr, l0$L0_te, time_tr, event_tr, meta_sn),
                 error=function(e) rowMeans(l0$L0_te, na.rm=TRUE))
      } else rowMeans(do.call(cbind, lapply(S$te, rowMeans, na.rm=TRUE)), na.rm=TRUE)
      rows[[length(rows)+1]] <<- rec(label, meta_sn, risk_se)
    }
    is.null(l0)   # TRUE = fallback
  }

  # Helper: ME-HAS (coxme) ek variant par; fallback flag ke saath
  run_has_variant <- function(S, label) {
    out <- tryCatch(
      run_has_surv(S$oof, S$te, time_tr, event_tr),
      error=function(e) {
        message(sprintf("  [FALLBACK] %s → rowMeans: %s", label, conditionMessage(e)))
        list(pred     = rowMeans(do.call(cbind, lapply(S$te, rowMeans, na.rm=TRUE))),
             varcomp  = list(var_slope_clf=NA_real_, var_intercept_imp=NA_real_),
             fallback = TRUE)
      })
    if (isTRUE(out$fallback))
      message(sprintf("  [FALLBACK] %s model fit fail -> standardised simple mean", label))
    rows[[length(rows)+1]] <<- rec(label, "All", out$pred)
    out
  }

  # ── (A) sirf MICE-PMM (Aleryani jaisa): MICE_SE = MAIN competitor; ME_HAS (PMM) ──
  #     MAIN_HAS_VARIANT = "MM" par yeh ME-HAS sirf supplementary mein ('ME_HAS_PMM')
  # Streams = classifier x PMM draw (draws ka average NAHI); same PMM
  # imputations aur OOF/test predictions jo MIE_PMM_AVG use karta hai.
  mehas_t0 <- proc.time()
  name_draws <- function(m) { colnames(m) <- paste0("pmm_d", seq_len(ncol(m))); m }
  S_pmm <- clean_streams(lapply(oof_list, name_draws), lapply(test_list, name_draws), "PMM")

  fb$mice_se_imp <- FALSE
  fb$mice_se_l0  <- run_se_variant(S_pmm, "MICE_SE")

  .META_ENV$variant <- "PMM"
  has_out <- run_has_variant(S_pmm, "ME_HAS")
  fb$me_has <- isTRUE(has_out$fallback)
  mehas_elapsed <- (proc.time() - mehas_t0)[["elapsed"]]

  # ── (B) 4 MICE methods (pmm, cart, norm, rf) -- MAIN ME-HAS jab MAIN_HAS_VARIANT = "MM";
  #     MICE_SE_MM sirf supplementary (same-imputation comparison) ──
  fb$me_has_mm <- NA
  if (RUN_MULTIMETHOD) {
    stack_imp <- list()
    for (mi in seq_along(MICE_STACK_METHODS)) {
      meth <- MICE_STACK_METHODS[mi]
      stack_imp[[meth]] <- if (meth == "pmm" && !isTRUE(fb$mice_na)) imp_tr
        else if (meth == "cart" && !isTRUE(fb$mice_cart)) imp_tr_cart
        else tryCatch(
          mice_impute_surv_method(X_tr_miss, time_tr, event_tr, meth,
                                  M_IMP, split_seed + 10L*mi, MAXIT_MICE),
          error=function(e) {
            message(sprintf("  [STACK MM] method '%s' imputation fail, chhoda: %s",
                            meth, conditionMessage(e)))
            NULL
          })
    }
    stack_imp <- Filter(Negate(is.null), stack_imp)
    if (length(stack_imp) > 0) {
      so <- list(); st <- list()
      for (sn in SURV_NAMES) {
        oc <- list(); tc <- list()
        for (meth in names(stack_imp)) {
          imps  <- stack_imp[[meth]]
          reuse <- meth == "pmm" && identical(imps, imp_tr)
          oof_m <- if (reuse) oof_list[[sn]] else
            tryCatch(get_oof_surv(sn, imps, time_tr, event_tr),
                     error=function(e) matrix(NA_real_, length(time_tr), length(imps)))
          te_m  <- if (reuse) test_list[[sn]] else
            run_mie_avg_surv(sn, imps, time_tr, event_tr,
                             lapply(seq_along(imps), function(i) X_te_full))$risk_mat
          oc[[meth]] <- rowMeans(oof_m, na.rm=TRUE)   # draws ka average per method
          tc[[meth]] <- rowMeans(te_m,  na.rm=TRUE)
        }
        so[[sn]] <- do.call(cbind, oc); st[[sn]] <- do.call(cbind, tc)
      }
      S_mm <- tryCatch(clean_streams(so, st, "MM"), error=function(e) {
        message(conditionMessage(e)); NULL })
      if (!is.null(S_mm)) {
        run_se_variant(S_mm, "MICE_SE_MM")
        .META_ENV$variant <- "MM"
        has_mm <- run_has_variant(S_mm, "ME_HAS_MM")
        fb$me_has_mm <- isTRUE(has_mm$fallback)
      }
    }
  }

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
    fb_me_has_mm   = fb$me_has_mm,
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
cat(sprintf("Models: %s\n", paste(SURV_NAMES, collapse=" / ")))
cat("Baselines: Complete(paired)/MeanSI/KNN/RFI/MissInd/MIE_PMM_AVG/MIE_CART_AVG/MICE_SE\n")
cat("Metrics: Harrell C-index | Brier(tau) | IBS(0..tau)\n")
cat(sprintf("Splits: %d | M_IMP: %d | MAXIT_MICE: %d\n", N_SPLITS, M_IMP, MAXIT_MICE))
cat("Checkpoints: split-level (dataset × mech × rate × split)\n\n")

all_results         <- list()
all_var_rows        <- list()
all_fb_rows         <- list()
all_miss_log        <- list()
all_time_rows       <- list()
all_mehas_time_rows <- list()
pipeline_t0         <- proc.time()

RESULT_COLS <- c("Method","Classifier","Cindex", CALIB_METRICS,
                 "Dataset","MissMech","MissRate","Split","ActualMiss")
FB_COLS     <- c("MissMech","MissRate",
                 "fb_knn","fb_rfi","fb_missind","fb_mice_na","fb_mice_cart",
                 "fb_mice_se_imp","fb_mice_se_l0","fb_me_has","fb_me_has_mm")

pad_df <- function(df, required_cols) {
  for (col in setdiff(required_cols, names(df))) df[[col]] <- NA_real_
  # Extra (purane version ke) columns bhi hata do aur order fix karo --
  # taa ke har data.frame ke columns bilkul same hon aur rbind kabhi fail na ho
  df[, required_cols, drop=FALSE]
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
      has_ibs         <- any(unlist(lapply(ckpt_raw$results,
                                           function(df) "IBS" %in% names(df))))
      methods_in_ckpt <- unique(unlist(lapply(ckpt_raw$results, function(df)
        if (!is.null(df$Method)) unique(df$Method) else character(0))))
      stale_methods   <- length(setdiff(METHOD_ORDER_ALL, methods_in_ckpt)) > 0 ||
        length(setdiff(methods_in_ckpt, METHOD_ORDER_ALL)) > 0
      clfs_in_ckpt    <- unique(unlist(lapply(ckpt_raw$results, function(df)
        if (!is.null(df$Classifier)) unique(df$Classifier) else character(0))))
      stale_clfs      <- length(setdiff(SURV_NAMES, clfs_in_ckpt)) > 0 ||
        length(setdiff(clfs_in_ckpt, c(SURV_NAMES,"All"))) > 0
      # FIX: ds-ckpt mein SAARE (mech x rate) cells aur har cell mein N_SPLITS
      # splits hone chahiye -- warna pilot / adhoora run poore run ki jagah load
      # ho jata (misal 2 splits wala pilot 10-split run mein skip ho jata).
      ck_all <- tryCatch(do.call(rbind, lapply(ckpt_raw$results, function(df)
        df[, intersect(c("MissMech","MissRate","Split","Method"), names(df)), drop=FALSE])),
        error=function(e) NULL)
      grid_incomplete <- TRUE
      if (!is.null(ck_all) && all(c("MissMech","MissRate","Split") %in% names(ck_all))) {
        ck_all <- ck_all[ck_all$Method == "ME_HAS", , drop=FALSE]
        grid_incomplete <- any(sapply(MISS_MECHS, function(mm) any(sapply(MISS_RATES, function(mr) {
          sp <- unique(ck_all$Split[ck_all$MissMech == mm & abs(as.numeric(ck_all$MissRate) - mr) < 1e-6])
          !all(seq_len(N_SPLITS) %in% sp)
        }))))
      }
      ds_stale <- length(missing_mechs)>0 || !has_ibs ||
        stale_methods || stale_clfs || grid_incomplete
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
        if (!has_ibs)      "missing IBS",
        if (grid_incomplete) "rates/splits incomplete",
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
  
  # (Purana alag complete-case loop hata diya: Complete ab run_one_surv() ke
  #  andar har missing cell ke SAME split par chalta hai -- paired reference.)

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
        # Split: har rate/mechanism par SAME (sirf split + dataset par depend)
        set.seed(make_partition_seed(sp, ds$name))
        tr_idx  <- caret::createDataPartition(DATASET$event, p=TRAIN_FRAC, list=FALSE)[, 1]
        te_idx  <- setdiff(seq_len(nrow(DATASET)), tr_idx)
        X_tr    <- DATASET[tr_idx, PRED_NAMES, drop=FALSE]
        time_tr <- DATASET[tr_idx, "time"]; evt_tr <- DATASET[tr_idx, "event"]
        X_te    <- DATASET[te_idx, PRED_NAMES, drop=FALSE]
        time_te <- DATASET[te_idx, "time"]; evt_te <- DATASET[te_idx, "event"]
        
        # Missingness mask: EK hi dafa, cell-specific seed se -- yahi mask
        # run_one_surv() ko jata hai
        sp_seed <- make_split_seed(sp, ds$name, miss_rate, mech)
        set.seed(sp_seed)
        X_tr_miss_cell <- tryCatch(
          induce_missing(X_tr, miss_rate, mech,
                         time_vec=time_tr, event_vec=evt_tr),
          error=function(e) X_tr)
        actual_miss_pct <- round(mean(is.na(X_tr_miss_cell)) * 100, 2)
        if (sp == 1) cat(sprintf("[actual=%.1f%%]", actual_miss_pct))
        
        miss_log_row <- data.frame(
          Dataset=ds$name, MissMech=mech, NominalRate=miss_rate,
          NominalPct=miss_rate*100, ActualMiss=actual_miss_pct/100,
          ActualPct=actual_miss_pct, Split=sp, stringsAsFactors=FALSE)
        ds_miss_log <- c(ds_miss_log, list(miss_log_row))
        
        split_t0 <- proc.time()
        .META_ENV$cell <- sprintf("%s|%s|%.0f%%|sp%02d", ds$name, mech, miss_rate*100, sp)
        out <- tryCatch(
          run_one_surv(mech, miss_rate, sp_seed,
                       X_tr, time_tr, evt_tr,
                       X_te, time_te, evt_te,
                       X_tr_miss = X_tr_miss_cell,
                       complete_seed = make_complete_seed(sp, ds$name)),
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
            long_format_rows=length(time_tr)*length(SURV_NAMES)*M_IMP,   # main ME-HAS: learners x PMM draws
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
                          elapsed_sec=round(pipeline_elapsed,3),
                          from_ckpt=NA,   # time_df jaisa hi 6th column, warna rbind fail
                          stringsAsFactors=FALSE)
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
# Reporting labels (dekhein MAIN_HAS_VARIANT): MM variant main "ME_HAS" ban jata hai
if (MAIN_HAS_VARIANT == "MM") {
  meth_chr <- as.character(combined$Method)
  stopifnot(any(meth_chr == "ME_HAS_MM"))
  meth_chr[meth_chr == "ME_HAS"]    <- "ME_HAS_PMM"
  meth_chr[meth_chr == "ME_HAS_MM"] <- "ME_HAS"
  combined$Method <- meth_chr
  cat("\n>>> REPORTING: ME_HAS = proposed ME-HAS (4 MICE methods: pmm, cart, norm, rf);",
      "ME_HAS_PMM = PMM-only reference. Saari tables, rank tables, Kendall's W aur figures",
      "isi labelling se bante hain (paper ke mutabiq).\n")
}
combined$Method   <- factor(combined$Method,   levels=REPORT_LEVELS)
stopifnot(!any(is.na(combined$Method)))
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
                 "fb_mice_se_imp","fb_mice_se_l0","fb_me_has","fb_me_has_mm")
  fb_labels <- c("KNN→MeanSI","RFI→MeanSI","MissInd→MeanSI","MICE+NA→stdMICE",
                 "CART→PMM","MICE_SE_imp→reuse","MICE_SE_L0→avg","ME_HAS→avg","ME_HAS_MM→avg")
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
    dplyr::across(dplyr::all_of(CALIB_METRICS),
                  list(mean=~round(mean(.x, na.rm=TRUE), 4),
                       sd  =~round(sd(.x,   na.rm=TRUE), 4))),
    .groups="drop")
write.csv(per_ds_summ,
          file.path(OUTPUT_DIR,"Multi_HAS_Survival_per_dataset_summary.csv"), row.names=FALSE)

grand_summ <- combined %>%
  dplyr::group_by(MissMech, MissRate, Method, Classifier) %>%
  dplyr::summarise(
    n_obs       = dplyr::n(),
    Cindex_mean = round(mean(Cindex, na.rm=TRUE), 4),
    Cindex_sd   = round(sd(Cindex,   na.rm=TRUE), 4),
    dplyr::across(dplyr::all_of(CALIB_METRICS),
                  list(mean=~round(mean(.x, na.rm=TRUE), 4),
                       sd  =~round(sd(.x,   na.rm=TRUE), 4))),
    .groups="drop")
write.csv(grand_summ,
          file.path(OUTPUT_DIR,"Multi_HAS_Survival_grand_summary.csv"), row.names=FALSE)
cat("Saved: summaries\n")

# ── 17. Method lists for rank tables ─────────────────────────────────────────
# (Wilcoxon / Bonferroni tests hata diye gaye -- paper mein sirf grand-mean
#  tables aur Kendall's W rank tables report hote hain.)
# Kendall rank tables mein yeh methods (har classifier ke saath) + ME_HAS.
# Complete reference hai, ranking mein shamil nahi.
RANK_METHODS <- c("MeanSI","KNN","RFI","MissInd","MIE_PMM_AVG","MIE_CART_AVG",
                  "MICE_SE","ME_HAS")

# ── 18. LaTeX Helpers ─────────────────────────────────────────────────────────
`%.%` <- function(a,b) paste0(a,b)

method_tex <- function(meth)
  switch(as.character(meth),
         ME_HAS       = "\\textbf{ME-HAS (proposed)}",
         MICE_SE      = "MICE-SE (Stacking)",
         ME_HAS_MM    = "ME-HAS (4 MICE methods)",
         ME_HAS_PMM   = "ME-HAS (PMM only)",
         MICE_SE_MM   = "MICE-SE (4 MICE methods)",
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
                              target_mech="MAR", method_levels=METHOD_ORDER) {
  mech_keys <- paste0(target_mech,
                      sprintf("%02d", as.integer(MISS_RATES*100)))
  present   <- intersect(c("COMP", mech_keys), names(wide))
  nc  <- sum("COMP" %in% present)
  nm2 <- sum(mech_keys %in% present)
  nd  <- nc + nm2
  wide <- wide %>%
    dplyr::filter(as.character(Method) %in% method_levels) %>%
    dplyr::mutate(Method=factor(Method, levels=method_levels)) %>%
    dplyr::arrange(Method, Classifier)
  hib <- if (metric %in% names(METRIC_HIB)) METRIC_HIB[[metric]] else TRUE
  best_row <- sapply(present, function(k) {
    vals <- unlist(wide[[k]])
    if (all(is.na(vals))) return(NA_integer_)
    best <- if (hib) max(vals, na.rm=TRUE) else min(vals, na.rm=TRUE)
    which(vals==best)[1]
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
    "MAR-type mechanism: logistic model on the row mean of the other standardised covariates, $\\log T$ and $\\Delta$, $\\beta=\\log(9)$ (OR$\\approx$9 per unit of this score)."
  else
    "MCAR mechanism: columns masked completely at random."
  metric_desc <- switch(metric,
    Cindex = "Harrell C-index (concordance; higher is better; \\texttt{survival::concordance}) \\citep{Harrell1996}.",
    Brier  = paste0("Brier score at $\\tau_{0.75}$ (75th percentile of test event times; lower is better;",
                    " IPCW, \\texttt{riskRegression::Score}) \\citep{Graf1999,Gerds2006}.",
                    " Risk scores mapped to probabilities by 5-fold cross-fitted Cox recalibration on the test set."),
    IBS    = paste0("Integrated Brier score over $[0,\\tau_{0.75}]$ (lower is better;",
                    " IPCW, \\texttt{riskRegression::Score}) \\citep{Graf1999,Gerds2006}.",
                    " Risk scores mapped to probabilities by 5-fold cross-fitted Cox recalibration on the test set."))
  imp_desc <- if (MAIN_HAS_VARIANT == "MM")
    paste0("MIE-PMM-Avg and MICE-SE: MICE-PMM ($m=5$); MIE-CART-Avg: MICE-CART ($m=5$);",
           " ME-HAS: four MICE methods (pmm, cart, norm, rf; $m=5$ each, averaged within method).")
  else "MIE-PMM-Avg, MICE-SE and ME-HAS: MICE-PMM ($m=5$); MIE-CART-Avg: MICE-CART ($m=5$)."
  cap <- sprintf(
    paste0("Grand mean %s ($\\text{mean}\\pm\\text{SD}$) averaged over %d",
           " right-censored survival benchmark datasets and %d stratified",
           " train--test splits (70/30). \\textit{Complete}: fully observed data on the",
           " same train--test split as each missingness cell (paired reference).",
           " \\textit{ME-HAS} (proposed): Mixed-Effects Hierarchical",
           " Aware Stacking. \\textit{Missing Indicator} (MissInd): binary",
           " missingness flags appended to mean-imputed data",
           " \\citep{Jones1996,Sperrin2020}. %s",
           " %s All MICE models include the Nelson--Aalen cumulative hazard and the event indicator",
           " \\citep{WhiteRoyston2009,vanBuurenGroothuis2011}."),
    metric_desc, length(DATASET_REGISTRY), N_SPLITS, mech_desc, imp_desc)
  build_latex_table(
    pivot_wide_ms(gs_m, metric), metric, cap,
    sprintf("tab:grand_%s_%s", tolower(metric), tolower(target_mech)),
    target_mech=target_mech)
}

# ── Kendall's W rank tables (binary pipeline jaisa) ─────────────────────────
# Har dataset ek "judge" (rater) hai jo saare method/classifier combinations ko
# rank karta hai (rank 1 = best). Kendall's W (tie-corrected, irr::kendall)
# batata hai ke datasets ranking par kitna agree karte hain (0 = koi agreement
# nahi, 1 = poora). chi2 = m (n-1) W, df = n-1 (Kendall & Babington Smith 1939).
compute_kendall_W <- function(rank_mat) {
  # Kendall's W seedha R package se: irr::kendall(ratings, correct = TRUE)
  #   ratings = objects (rows: method/classifier) x raters (columns: datasets)
  #   correct = TRUE -> ties ke liye correction (Wt)
  #   irr::kendall har column ko khud rank() karta hai; hamare ranks (1 = best)
  #   par rank() wahi order deta hai, isliye nateeja same hai.
  #   Note: irr::kendall na.omit() karta hai -- jis row mein kisi dataset ka NA
  #   ho woh hat jati hai (n_objects output mein dikhta hai).
  if (nrow(rank_mat) < 2 || ncol(rank_mat) < 2)
    return(list(W = NA_real_, chi2 = NA_real_, df = NA_integer_,
                p_value = NA_real_, n_objects = nrow(rank_mat)))
  kw <- irr::kendall(rank_mat, correct = TRUE)
  list(W         = round(unname(kw$value), 4),
       chi2      = round(unname(kw$statistic), 3),
       df        = as.integer(kw$subjects - 1L),
       p_value   = signif(unname(kw$p.value), 4),
       n_objects = kw$subjects)
}

build_rank_table <- function(data_in, miss_rate_val, metric = "Cindex",
                             higher_is_better = TRUE, mech = "MAR") {
  sub <- data_in %>%
    dplyr::filter(Method != "Complete",            # Complete reference hai, competitor nahi
                  Method %in% RANK_METHODS,
                  abs(MissRate - miss_rate_val) < 1e-6)
  if (nrow(sub) == 0) {
    message(sprintf("[Kendall] No %s data at %g%%", mech, miss_rate_val * 100))
    return(NULL)
  }
  # Har dataset ke liye split-average value
  agg <- sub %>%
    dplyr::group_by(Method, Classifier, Dataset) %>%
    dplyr::summarise(metric_mean = mean(.data[[metric]], na.rm = TRUE), .groups = "drop") %>%
    dplyr::mutate(MethodClf = paste(Method, Classifier, sep = "/"))
  row_order <- c(unlist(lapply(setdiff(RANK_METHODS, ALL_CLF_METHODS),
                               function(m) paste(m, SURV_NAMES, sep = "/"))),
                 paste(ALL_CLF_METHODS, "All", sep = "/"))
  row_order <- row_order[row_order %in% unique(agg$MethodClf)]

  wide_val <- agg %>%
    dplyr::select(MethodClf, Dataset, metric_mean) %>%
    tidyr::pivot_wider(names_from = Dataset, values_from = metric_mean) %>%
    dplyr::mutate(MethodClf = factor(MethodClf, levels = row_order)) %>%
    dplyr::arrange(MethodClf)
  ds_cols <- setdiff(names(wide_val), "MethodClf")

  # Har dataset mein rank (1 = best). C-index: zyada behtar; Brier/IBS: kam behtar.
  sign_flip <- if (higher_is_better) -1 else 1
  rank_wide <- wide_val
  for (dc in ds_cols)
    rank_wide[[dc]] <- rank(sign_flip * wide_val[[dc]],
                            ties.method = "average", na.last = "keep")
  rank_mat <- as.matrix(rank_wide[, ds_cols, drop = FALSE])
  rownames(rank_mat) <- as.character(rank_wide$MethodClf)
  rank_wide$MeanRank    <- round(rowMeans(rank_mat, na.rm = TRUE), 2)
  rank_wide$OverallRank <- rank(rank_wide$MeanRank, ties.method = "average")
  rank_wide <- rank_wide[order(rank_wide$OverallRank), ]
  rank_mat  <- as.matrix(rank_wide[, ds_cols, drop = FALSE])
  rownames(rank_mat) <- as.character(rank_wide$MethodClf)

  list(miss_rate = miss_rate_val, metric = metric, mech = mech,
       val_wide = wide_val, rank_wide = rank_wide, rank_mat = rank_mat,
       datasets = ds_cols, kendall = compute_kendall_W(rank_mat))
}

rank_table_to_latex <- function(rt) {
  mr_pct  <- as.integer(round(rt$miss_rate * 100))
  met     <- rt$metric
  rw      <- rt$rank_wide
  ds_cols <- rt$datasets
  kw      <- rt$kendall
  n_ds    <- length(ds_cols)
  n_cols  <- n_ds + 2L
  p_txt   <- ifelse(is.na(kw$p_value), "\\text{NA}",
                    ifelse(kw$p_value < 0.001, "<0.001", sprintf("%.4f", kw$p_value)))
  cap <- sprintf(paste0(
    "Dataset-level rankings of method--model combinations (\\textbf{%s}, %s; %d\\%% %s). ",
    "Rank~1 denotes the best combination on each dataset; rows are sorted by mean rank ",
    "across %d right-censored survival datasets. Concordance of rankings across datasets: ",
    "Kendall's $W = %.4f$, $\\chi^2(%d) = %.3f$, $p = %s$."),
    METRIC_LABEL[[met]], if (METRIC_HIB[[met]]) "higher is better" else "lower is better",
    mr_pct, rt$mech, n_ds,
    ifelse(is.na(kw$W), NA, kw$W), kw$df, ifelse(is.na(kw$chi2), NA, kw$chi2), p_txt)
  hdr <- paste(c(sprintf("\\textbf{\\rotatebox{45}{%s}}", gsub("_", " ", ds_cols)),
                 "\\textbf{Mean Rank}", "\\textbf{Overall Rank}"), collapse = " & ")
  lines <- c("\\begin{table}[!ht]", "  \\centering",
             sprintf("  \\caption{%s}", cap),
             sprintf("  \\label{tab:rank_%s_%s%02d}", tolower(met), rt$mech, mr_pct),
             "  \\resizebox{\\linewidth}{!}{%",
             sprintf("  \\begin{tabular}{@{}l%s@{}}", strrep("c", n_cols)),
             "  \\toprule",
             sprintf("  \\textbf{Method/Model} & %s \\\\", hdr),
             "  \\midrule")
  fmt_r <- function(v) if (is.na(v)) "--" else if (v == floor(v)) as.character(as.integer(v)) else sprintf("%.1f", v)
  for (i in seq_len(nrow(rw))) {
    parts <- strsplit(as.character(rw$MethodClf[i]), "/")[[1]]
    ds_cells <- sapply(ds_cols, function(dc) {
      rv <- rw[[dc]][i]; s <- fmt_r(rv)
      if (!is.na(rv) && rv <= 1.5) sprintf("\\textbf{%s}", s) else s
    })
    mr_fmt <- sprintf("%.2f", rw$MeanRank[i])
    if (!is.na(rw$MeanRank[i]) && rw$MeanRank[i] == min(rw$MeanRank, na.rm = TRUE))
      mr_fmt <- sprintf("\\textbf{%s}", mr_fmt)
    or_fmt <- fmt_r(rw$OverallRank[i])
    if (!is.na(rw$OverallRank[i]) && rw$OverallRank[i] <= 1.5)
      or_fmt <- sprintf("\\textbf{%s}", or_fmt)
    row_label <- sprintf("%s/%s", method_tex(parts[1]), parts[2])
    lines <- c(lines, sprintf("  %s & %s \\\\", row_label,
                              paste(c(ds_cells, mr_fmt, or_fmt), collapse = " & ")))
  }
  lines <- c(lines, "  \\bottomrule",
             sprintf("  \\multicolumn{%d}{l}{\\footnotesize Kendall's $W = %.4f$; $\\chi^2 = %.3f$; $df = %d$; $p = %s$} \\\\",
                     n_cols + 1L, ifelse(is.na(kw$W), 0, kw$W),
                     ifelse(is.na(kw$chi2), 0, kw$chi2), kw$df, p_txt),
             "  \\end{tabular}%", "  }", "\\end{table}", "")
  paste(lines, collapse = "\n")
}

# ME-HAS coxme diagnostics (har session ki part files + purani single file)
load_mehas_diag <- function() {
  parts <- list.files(OUTPUT_DIR, pattern="^Multi_HAS_Survival_mehas_diagnostics_part_.*\\.csv$",
                      full.names=TRUE)
  old_f <- file.path(OUTPUT_DIR, "Multi_HAS_Survival_mehas_diagnostics.csv")
  if (file.exists(old_f)) parts <- c(old_f, parts)
  if (length(parts) == 0) return(NULL)
  md <- dplyr::bind_rows(lapply(parts, function(f)
    tryCatch(read.csv(f, stringsAsFactors=FALSE), error=function(e) NULL)))
  if (nrow(md) == 0) return(NULL)
  if (!"Variant" %in% names(md)) md$Variant <- "PMM"
  md <- md[sub("\\|.*", "", md$Cell) %in% sapply(DATASET_REGISTRY, `[[`, "name"), , drop=FALSE]
  md[!duplicated(paste(md$Cell, md$Variant), fromLast=TRUE), , drop=FALSE]
}

make_varcomp_table <- function(target_mech="MAR") {
  # Variance components MAIN ME-HAS variant ke coxme fits se (diagnostics file).
  md <- load_mehas_diag()
  if (is.null(md)) return("")
  md <- md[md$Variant == MAIN_HAS_VARIANT, , drop=FALSE]
  cell_parts <- strsplit(md$Cell, "|", fixed=TRUE)
  md$Dataset  <- sapply(cell_parts, `[`, 1)
  md$MissMech <- sapply(cell_parts, `[`, 2)
  md$MissRate <- as.numeric(sub("%", "", sapply(cell_parts, `[`, 3))) / 100
  vc <- md %>%
    dplyr::filter(MissMech==target_mech) %>%
    dplyr::group_by(Dataset, MissRate) %>%
    dplyr::summarise(slope_mean =mean(var_slope_clf,        na.rm=TRUE),
                     interc_mean=mean(var_intercept_method, na.rm=TRUE),
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
      " ($1\\mid\\texttt{method\\_id}$; %s).",
      " Extracted via \\texttt{coxme::VarCorr} \\citep{Therneau2012}.}"),
      N_SPLITS, target_mech,
      if (MAIN_HAS_VARIANT == "MM") "between-imputation-method shift: pmm, cart, norm, rf"
      else "between-PMM-draw shift"),
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
  for (met in EVAL_METRICS)
    gt_lines <- c(gt_lines, make_grand_table(grand_summ, met, target_mech=mech))
  gt_lines <- c(gt_lines, "\\end{document}")
  writeLines(gt_lines, grand_tex)
  cat(sprintf("Saved: %s\n", basename(grand_tex)))
  
  vc_tex   <- file.path(OUTPUT_DIR,
                        sprintf("Multi_HAS_Survival_varcomp_table_%s.tex", mech))
  vc_lines <- c(latex_preamble(10), make_varcomp_table(mech), "\\end{document}")
  writeLines(vc_lines, vc_tex)
  cat(sprintf("Saved: %s\n", basename(vc_tex)))
  
}

# ── SUPPLEMENTARY: PMM vs 4-method stacking (MICE_SE, ME-HAS) ───────────────
# Same splits, same masks; sirf stacking ki imputation alag (PMM vs pmm/cart/norm/rf).
if (RUN_MULTIMETHOD) {
  # Reporting labels (MAIN_HAS_VARIANT = "MM"): ME_HAS = 4-method, ME_HAS_PMM = PMM-only
  if (MAIN_HAS_VARIANT == "MM") {
    SUPP_TABLE_METHODS <- c("MICE_SE", "MICE_SE_MM", "ME_HAS_PMM", "ME_HAS")
    MM_LABELS          <- c("MICE_SE_MM", "ME_HAS")
  } else {
    SUPP_TABLE_METHODS <- c("MICE_SE", "MICE_SE_MM", "ME_HAS", "ME_HAS_MM")
    MM_LABELS          <- c("MICE_SE_MM", "ME_HAS_MM")
  }
  supp_csv <- grand_summ %>%
    dplyr::filter(as.character(Method) %in% SUPP_TABLE_METHODS) %>%
    dplyr::mutate(Imputation = ifelse(as.character(Method) %in% MM_LABELS,
                                      "MICE: pmm+cart+norm+rf (4 x 5)", "MICE-PMM (m=5)"))
  write.csv(supp_csv, file.path(OUTPUT_DIR, "Multi_HAS_Survival_supp_multimethod.csv"),
            row.names=FALSE)
  for (mech in MISS_MECHS) {
    supp_lines <- latex_preamble(10)
    for (met in EVAL_METRICS) {
      gs_m <- grand_summ %>%
        dplyr::filter(MissMech == mech, as.character(Method) %in% SUPP_TABLE_METHODS)
      cap <- sprintf(paste0(
        "Supplementary: stacking with MICE-PMM ($m=5$) versus four MICE",
        " imputation methods (pmm, cart, norm, rf; 5 draws each) -- %s, %s.",
        " Main analysis: MICE-SE with MICE-PMM, ME-HAS with four methods.",
        " Same train--test splits and missingness masks as the main analysis;",
        " for each variant, MICE-SE and ME-HAS use identical level-0 predictions."),
        METRIC_LABEL[[met]], mech)
      supp_lines <- c(supp_lines, build_latex_table(
        pivot_wide_ms(gs_m, met), met, cap,
        sprintf("tab:supp_mm_%s_%s", tolower(met), tolower(mech)),
        target_mech = mech, method_levels = SUPP_TABLE_METHODS))
    }
    supp_lines <- c(supp_lines, "\\end{document}")
    writeLines(supp_lines, file.path(OUTPUT_DIR,
               sprintf("Multi_HAS_Survival_supp_multimethod_%s.tex", mech)))
    cat(sprintf("Saved: Multi_HAS_Survival_supp_multimethod_%s.tex\n", mech))
  }
}

# ── Kendall's W rank tables: har mechanism x metric x missingness rate ───────
cat("\n=== Kendall's W Rank Tables (C-index, Brier, IBS) ===\n")
rank_table_list      <- list()
kendall_summary_rows <- list()
for (mech in MISS_MECHS) {
  mech_data <- combined %>% dplyr::filter(MissMech == mech)
  for (met in EVAL_METRICS) {
    hib <- METRIC_HIB[[met]]
    tex_lines <- c(latex_preamble(9),
                   sprintf("%% Kendall's W Rank Tables -- %s, %s, one per missingness level", mech, met), "")
    for (mr in MISS_RATES) {
      mr_pct <- as.integer(round(mr * 100))
      rt <- build_rank_table(mech_data, mr, metric = met,
                             higher_is_better = hib, mech = mech)
      if (is.null(rt)) next
      rank_table_list[[paste0(mech, "_", met, "_", mr_pct)]] <- rt
      kw <- rt$kendall
      cat(sprintf("%s | %-6s | %2d%%: W = %.4f, chi2(%d) = %.3f, p = %s\n",
                  mech, met, mr_pct, ifelse(is.na(kw$W), NaN, kw$W), kw$df,
                  ifelse(is.na(kw$chi2), NaN, kw$chi2),
                  ifelse(is.na(kw$p_value), "NA",
                         ifelse(kw$p_value < 0.001, "< 0.001", sprintf("%.4f", kw$p_value)))))
      kendall_summary_rows[[length(kendall_summary_rows) + 1]] <- data.frame(
        Metric = met, MissMech = mech, MissRate = mr, MissRate_pct = mr_pct,
        W = kw$W, chi2 = kw$chi2, df = kw$df, p_value = kw$p_value,
        n_methods = kw$n_objects, n_datasets = ncol(rt$rank_mat),
        stringsAsFactors = FALSE)
      csv_rt <- rt$rank_wide
      csv_rt$MissRate_pct <- mr_pct; csv_rt$Metric <- met; csv_rt$MissMech <- mech
      write.csv(csv_rt, file.path(OUTPUT_DIR,
                sprintf("Multi_HAS_Survival_rank_table_%s_%02dpct_%s.csv", mech, mr_pct, met)),
                row.names = FALSE)
      tex_lines <- c(tex_lines, rank_table_to_latex(rt))
    }
    tex_lines <- c(tex_lines, "\\end{document}")
    writeLines(tex_lines, file.path(OUTPUT_DIR,
               sprintf("Multi_HAS_Survival_rank_tables_%s_%s.tex", mech, met)))
  }
}
if (length(kendall_summary_rows) > 0) {
  kendall_df <- do.call(rbind, kendall_summary_rows); rownames(kendall_df) <- NULL
  write.csv(kendall_df, file.path(OUTPUT_DIR, "Multi_HAS_Survival_kendall_W_summary.csv"),
            row.names = FALSE)
  cat("Saved: Multi_HAS_Survival_kendall_W_summary.csv\n")
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
surv_shape_pal <- setNames(c(16L, 17L, 15L, 18L, 8L, 4L)[seq_along(SURV_NAMES)], SURV_NAMES)
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
                     Mean_Brier  = mean(Brier,  na.rm=TRUE),
                     Mean_IBS    = mean(IBS,    na.rm=TRUE),
                     .groups="drop")
  
  mehas_raw <- clf_raw %>% dplyr::filter(Method %in% ALL_CLF_METHODS, Classifier=="All")
  mehas_rep <- purrr::map_dfr(SURV_NAMES, ~dplyr::mutate(mehas_raw, Classifier=.x))
  
  plot_clf <- dplyr::bind_rows(
    clf_raw %>% dplyr::filter(!Method %in% ALL_CLF_METHODS),
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
  
  # Fig A2/A3: Grand mean Brier aur IBS (kam = behtar)
  for (fm in list(list(col="Mean_Brier", lab="Mean Brier score at tau (lower is better)",
                       tag="figA2_grand_brier", ttl="Brier Score"),
                  list(col="Mean_IBS",   lab="Mean integrated Brier score (lower is better)",
                       tag="figA3_grand_ibs",   ttl="Integrated Brier Score"))) {
    pX <- make_traj_plot(
      fm$col, fm$lab,
      sprintf("Grand Mean %s Trajectories Under %s", fm$ttl, mech),
      sprintf("Grand mean over %d benchmark datasets. tau = 75th percentile of test event times; IPCW (riskRegression::Score).",
              length(unique(combined$Dataset))))
    ggsave(file.path(OUTPUT_DIR, sprintf("%s_%s.png", fm$tag, mech)),
           plot=pX, width=16, height=5, units="in", dpi=900, bg="white")
    ggsave(file.path(OUTPUT_DIR, sprintf("%s_%s.pdf", fm$tag, mech)),
           plot=pX, width=16, height=5, units="in", device=cairo_pdf)
    cat(sprintf("Saved: %s_%s (.png + .pdf)\n", fm$tag, mech))
  }
  
  # Fig B: Per-dataset C-index
  clf_ds_raw <- combined %>%
    dplyr::filter(Method %in% METHOD_ORDER, MissMech==mech) %>%
    dplyr::group_by(MissRate, Method, Classifier, Dataset) %>%
    dplyr::summarise(Mean_Cindex=mean(Cindex, na.rm=TRUE), .groups="drop")
  mehas_ds_raw <- clf_ds_raw %>%
    dplyr::filter(Method %in% ALL_CLF_METHODS, Classifier=="All")
  mehas_ds_rep <- purrr::map_dfr(SURV_NAMES,
                                 ~dplyr::mutate(mehas_ds_raw, Classifier=.x))
  plot_clf_ds <- dplyr::bind_rows(
    clf_ds_raw %>% dplyr::filter(!Method %in% ALL_CLF_METHODS),
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
cat("\n=== METHOD-AVERAGED C-index / Brier / IBS @ 50% missingness ===\n")
cat("(ME_HAS: All classifiers; others: averaged across classifiers)\n")
grand_summ %>%
  dplyr::filter(abs(MissRate-0.50)<1e-6, as.character(Method) %in% METHOD_ORDER) %>%
  dplyr::group_by(MissMech, Method) %>%
  dplyr::summarise(Cindex_mean = round(mean(Cindex_mean, na.rm=TRUE), 4),
                   Brier_mean  = round(mean(Brier_mean,  na.rm=TRUE), 4),
                   IBS_mean    = round(mean(IBS_mean,    na.rm=TRUE), 4),
                   .groups="drop") %>%
  dplyr::mutate(Method=factor(Method, levels=METHOD_ORDER)) %>%
  dplyr::arrange(MissMech, dplyr::desc(Cindex_mean)) %>%
  as.data.frame() %>% print(row.names=FALSE)

cat("\n=== PER-CLASSIFIER breakdown @ 50% missingness ===\n")
grand_summ %>%
  dplyr::filter(abs(MissRate-0.50)<1e-6, as.character(Method) %in% METHOD_ORDER) %>%
  dplyr::mutate(Method=factor(Method, levels=METHOD_ORDER)) %>%
  dplyr::arrange(MissMech, Method, Classifier) %>%
  dplyr::select(MissMech, Method, Classifier, Cindex_mean, Brier_mean, IBS_mean) %>%
  as.data.frame() %>% print(row.names=FALSE)

# ── 20b. ME-HAS (coxme) diagnostics summary ─────────────────────────────────
# Har variant (PMM = main, MM = supplementary) ke liye alag: kitne fits full
# model ke saath hue, kitne fallback par gaye, aur singular / near-boundary rates.
diag_parts <- list.files(OUTPUT_DIR, pattern="^Multi_HAS_Survival_mehas_diagnostics_part_.*\\.csv$",
                         full.names=TRUE)
# Purane runs ki single diagnostics file bhi shamil karo (backward compatible)
old_diag <- file.path(OUTPUT_DIR, "Multi_HAS_Survival_mehas_diagnostics.csv")
if (file.exists(old_diag)) diag_parts <- c(old_diag, diag_parts)
if (length(diag_parts) > 0) {
  md <- dplyr::bind_rows(lapply(diag_parts, function(f)
    tryCatch(read.csv(f, stringsAsFactors=FALSE), error=function(e) NULL)))
  # Sirf is run ke datasets/mechanisms ke cells rakho
  md <- md[sub("\\|.*", "", md$Cell) %in% sapply(DATASET_REGISTRY, `[[`, "name"), , drop=FALSE]
  if (!"Variant" %in% names(md)) md$Variant <- "PMM"
  md <- md[!duplicated(paste(md$Cell, md$Variant), fromLast=TRUE), ]
  write.csv(md, file.path(OUTPUT_DIR, "Multi_HAS_Survival_mehas_diagnostics_dedup.csv"),
            row.names=FALSE)
  for (vv in unique(md$Variant)) {
    m2  <- md[md$Variant == vv, , drop=FALSE]
    pct <- function(x) sprintf("%d/%d (%.1f%%)", sum(x, na.rm=TRUE), nrow(m2),
                               100*mean(x, na.rm=TRUE))
    cat(sprintf("\n=== ME-HAS coxme diagnostics [%s] ===\n",
                paste0(if (vv == MAIN_HAS_VARIANT) "MAIN: " else "SUPP: ",
                       if (vv == "PMM") "PMM only" else "4 MICE methods")))
    cat(sprintf("Total fits                 : %d\n", nrow(m2)))
    cat(sprintf("Full model                 : %s\n", pct(m2$full_model %in% TRUE)))
    cat(sprintf("Fallback simple mean       : %s\n", pct(grepl("^NONE", m2$formula))))
    if ("singular_clf" %in% names(m2)) {
      cat("Singular = SD < 1e-4 (lme4::isSingular jaisa); near = SD < 0.01\n")
      cat(sprintf("Classifier slope   singular / near: %s / %s\n",
                  pct(m2$singular_clf %in% TRUE), pct(m2$near_clf %in% TRUE)))
      cat(sprintf("%-18s singular / near: %s / %s\n",
                  if (vv == "PMM") "Draw intercept" else "Method intercept",
                  pct(m2$singular_method %in% TRUE), pct(m2$near_method %in% TRUE)))
    }
  }
}

# ── 20c. Learner failure report ─────────────────────────────────────────────
lrn <- union(names(.FIT_LOG$n), names(.FIT_LOG$fb))
if (length(lrn) > 0) {
  lf <- data.frame(Learner = lrn,
                   n_errors    = sapply(lrn, function(l) if (is.null(.FIT_LOG$n[[l]])) 0 else .FIT_LOG$n[[l]]),
                   n_fallbacks = sapply(lrn, function(l) if (is.null(.FIT_LOG$fb[[l]])) 0 else .FIT_LOG$fb[[l]]),
                   example_error = sapply(lrn, function(l) paste(unique(.FIT_LOG$msg[[l]]), collapse=" || ")),
                   stringsAsFactors = FALSE, row.names = NULL)
  write.csv(lf, file.path(OUTPUT_DIR, sprintf("Multi_HAS_Survival_learner_failures_%s.csv", SESSION_TAG)),
            row.names=FALSE)
  cat("\n=== Learner failures (is run mein; checkpoint se load hue cells shamil nahi) ===\n")
  print(lf[, c("Learner","n_errors","n_fallbacks")], row.names=FALSE)
} else cat("\n=== Learner failures: koi nahi ===\n")

# ── 21. Final Summary ─────────────────────────────────────────────────────────
cat("\n=============================================\n")
cat("  ALL DONE — ME-HAS Survival Benchmark\n")
cat("=============================================\n")
cat("Methods: Complete(paired)/MeanSI/KNN/RFI/MissInd/MIE_PMM_AVG/MIE_CART_AVG/MICE_SE/ME_HAS\n")
cat(sprintf("Main ME-HAS variant: %s | MICE_SE (main): PMM only\n",
            if (MAIN_HAS_VARIANT == "MM") "4 MICE methods (pmm+cart+norm+rf)" else "PMM only"))
cat("Metrics: Harrell C (survival::concordance) | Brier(tau), IBS(0..tau) via riskRegression::Score\n")
cat("Kendall's W: irr::kendall(correct=TRUE) on per-dataset ranks for C-index, Brier, IBS\n")
cat("Tables: grand mean (C-index, Brier, IBS) + Kendall's W rank tables\n")
cat(sprintf("\nOutput: %s/\n", OUTPUT_DIR))

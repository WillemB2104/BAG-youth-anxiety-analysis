#############################################################
# Brain Age Gap (BAG) Analysis 
# Author:  Willem Bruin
#
# Description:
# Processes FreeSurfer-derived MRI features and MCCQRNN model 
# predictions to compute BAG metrics, perform group-level 
# analyses, generate plots, and conduct occlusion sensitivity 
# mapping.
#
# Main steps:
#   1. Load and preprocess input data
#   2. Fit linear mixed-effects models for group comparisons
#   3. Perform transdiagnostic and subgroup analyses
#   4. Test within-patient clinical associations
#   5. Conduct occlusion sensitivity mapping to identify 
#      region-specific contributions to BAG
#
# Inputs:
#   - FreeSurfer morphometric measures (regional + global)
#   - MCCQRNN brain age predictions
#   - Clinical/demographic variables
#
# Outputs:
#   - Excel files with model results and multiple comparison 
#     corrections
#   - Plots of BAG distributions and effects
#   - Occlusion sensitivity maps per group, age band, and sex
#############################################################


###########################
# 1. Load Required Packages
###########################
library(tidyverse)
library(ggplot2)
library(lme4)
library(nlme)
library(afex)
library(emmeans)
library(sjPlot)
library(sjmisc)
library(sjlabelled)
library(ggeffects)
library(ggseg)
library(xlsx)
library(effects)


###################################
# 2. Set Working & Data Directories
###################################
mainDir <- "<PATH_TO_MAIN_DIRECTORY>"
setwd(mainDir)
dataDir <- file.path(mainDir, "data")


#############################
# 3. Import Custom Functions
#############################
source("scripts/exclude_subjects_with_missing_features.R")
source("scripts/brain_age_mapping_NAKO_LME4.R")
source("scripts/lme_helper_functions.R")


##########################
# 4. Import Data
##########################
pooled_df_no_duplicates <- read.csv(
  file.path(dataDir, "enigma", "cross_disorder_data_v3", 
            "pooled_cross_disorder_data_no_duplicates.csv"),
  na.strings = c("", "NA"), sep = ",", dec = "."
)
MCC_results_HC_df <- read.csv(
  file.path(dataDir, "mccqrnn", "predictions_HC.csv"),
  na.strings = c("", "NA"), sep = ",", dec = "."
)
MCC_results_PT_df <- read.csv(
  file.path(dataDir, "mccqrnn", "predictions_patients.csv"),
  na.strings = c("", "NA"), sep = ",", dec = "."
)


######################################
# 5. FreeSurfer Data Cleanup & Filtering
######################################
FS_indices <- which(colnames(pooled_df_no_duplicates) == "L_bankssts_surfavg"):
  which(colnames(pooled_df_no_duplicates) == "ICV")
FS_labels <- colnames(pooled_df_no_duplicates)[FS_indices]
global_FS_features <- c("LSurfArea", "RSurfArea", "LThickness", "RThickness", "ICV")
FS_labels_wo_global <- FS_labels[!(FS_labels %in% global_FS_features)]
ct_mask <- grepl("thick", FS_labels_wo_global)
csa_mask <- grepl("surf", FS_labels_wo_global)
subcort_mask <- !(ct_mask | csa_mask)

# Age filter
pooled_df_no_duplicates <- pooled_df_no_duplicates %>%
  filter(Age >= 10 & Age <= 25)

# Remove incomplete FS subjects
pooled_df_no_duplicates <- exclude_subjects_with_missing_features(
  pooled_df_no_duplicates,
  FS_labels_wo_global,
  completeness_threshold = 0.75
)


##########################
# 6. Sample Size Overview
##########################
print(aggregate(MultiSiteID ~ WG, data = pooled_df_no_duplicates, FUN = function(x) length(unique(x))))
print(table(pooled_df_no_duplicates[c("WG", "Dx")]))


################################
# 7. Extract HC & Patient Datasets
################################
HC_df <- filter(pooled_df_no_duplicates, Dx == 0)
PT_df <- filter(pooled_df_no_duplicates, Dx == 1)


########################################
# 8. Compute Adjusted & Unadjusted BAGs
########################################
MCC_results_PT_df$y_pred <- MCC_results_PT_df$median_aleatory_epistemic
MCC_results_HC_df$y_pred <- MCC_results_HC_df$median_aleatory_epistemic
adj_BAG_PT <- (MCC_results_PT_df$y_pred - MCC_results_PT_df$y_true) / MCC_results_PT_df$std_aleatory_epistemic
adj_BAG_HC <- (MCC_results_HC_df$y_pred - MCC_results_HC_df$y_true) / MCC_results_HC_df$std_aleatory_epistemic
unadj_BAG_PT <- MCC_results_PT_df$y_pred - MCC_results_PT_df$y_true
unadj_BAG_HC <- MCC_results_HC_df$y_pred - MCC_results_HC_df$y_true

stopifnot(all.equal(adj_BAG_PT, MCC_results_PT_df$adj_BAG))
stopifnot(all.equal(adj_BAG_HC, MCC_results_HC_df$adj_BAG))
stopifnot(all.equal(HC_df$Age, MCC_results_HC_df$y_true))
stopifnot(all.equal(PT_df$Age, MCC_results_PT_df$y_true))


###########################################
# 9. Merge BAGs with Clinical Information
###########################################
BAG_per_group_df <- PT_df %>%
  select(WG, Dx, Age, Sex, MultiSiteID, SubjID, MDD, STAI_T, AgeO, Med) %>%
  mutate(WG_HC = WG,
         adj_BAG = adj_BAG_PT,
         unadj_BAG = unadj_BAG_PT)
HC_df_subset <- HC_df %>%
  select(WG, Dx, Age, Sex, MultiSiteID, SubjID, MDD, STAI_T, AgeO, Med) %>%
  rename(WG_HC = WG)
BAG_per_group_df <- rbind(
  BAG_per_group_df,
  cbind(WG = "HC", HC_df_subset,
        adj_BAG = adj_BAG_HC,
        unadj_BAG = unadj_BAG_HC)
)
BAG_per_group_df <- BAG_per_group_df %>%
  mutate(
    adj_BAG = as.numeric(adj_BAG),
    unadj_BAG = as.numeric(unadj_BAG),
    Age = as.numeric(Age),
    Sex = as.factor(Sex),
    WG = as.factor(WG)
  )
BAG_per_group_df$STAI_T[BAG_per_group_df$STAI_T < 20 | BAG_per_group_df$STAI_T > 80] <- NA

# Parse median STAI_T & AgeO
median_AgeO <- median(PT_df$AgeO, na.rm = TRUE)
median_STAI_T <- median(PT_df$STAI_T, na.rm = TRUE)
cat("Median Age of Onset: "); print(median_AgeO)
cat("Median STAI_T: "); print(median_STAI_T)


#############################
# 10. Analysis Parameters
#############################
MCP_method <- "fdr"
MCP_alpha <- 0.05
BAG_per_group_df <- within(BAG_per_group_df, WG <- relevel(WG, ref = "HC"))
ctrl <- lmeControl(opt = "optim")


###################################
# 11. Main BAG Analyses
###################################

##------------------------------------------------
## Step 1: Regress out covariates and plot results
##------------------------------------------------

# Fit linear mixed-effects model to adjust BAG for covariates
plot_res <- lme(
  adj_BAG ~ Age + I(Age^2) + Sex,
  random = ~ 1 | MultiSiteID,   # Random intercept for each site
  method = "ML",
  na.action = na.exclude,
  control = list(opt = "optim"),
  data = BAG_per_group_df
)

# Store model residuals for plotting
BAG_per_group_df$plot_res <- resid(plot_res)

# Compute group means for plotting reference lines
mean_values <- aggregate(plot_res ~ Dx, data = BAG_per_group_df, FUN = mean)

# Create density plot with vertical mean lines and rug marks
ggplot(BAG_per_group_df, aes(x = plot_res, color = as.factor(Dx), fill = as.factor(Dx))) +
  geom_density(alpha = 0.4) +
  geom_vline(data = mean_values, aes(xintercept = plot_res, color = as.factor(Dx)),
             linetype = "dashed", size = 0.8) +
  geom_rug(aes(color = as.factor(Dx)), sides = "b") +
  scale_color_manual(values = c("black", "red"), labels = c("Control", "Patient")) +
  scale_fill_manual(values = c("black", "red"), labels = c("Control", "Patient")) +
  labs(
    title = "Case-control differences in brain-aging",
    x = "Brain–Age Gap",
    y = "Density",
    color = "Group",
    fill = "Group"
  )

# Save residual data for further use
write.csv(BAG_per_group_df, "BAG_residuals_per_group.csv", row.names = FALSE)


##------------------------------------------------
## Step 2: Loop through BAG types and run analyses
##------------------------------------------------

BAG_types <- c("adj_BAG")  # Optionally: c("adj_BAG", "unadj_BAG")

for (BAG_type in BAG_types) {
  cat("Running analyses for:", BAG_type, "\n")
  
  # Set BAG column for this iteration
  BAG_per_group_df$BAG <- BAG_per_group_df[[BAG_type]]
  
  # Create results directory if it doesn't exist
  resultsDir <- file.path(mainDir, paste("lme_results", BAG_type, sep = "_"))
  if (!file.exists(resultsDir)) dir.create(resultsDir)
  
  # Initialize results file
  lmeresultsPath <- file.path(resultsDir, "lmeResults.xlsx")
  if (file.exists(lmeresultsPath)) file.remove(lmeresultsPath)
  
  ###------------------------------------------------
  ### Step 2.1: Main case-control analysis
  ###------------------------------------------------
  
  # Fit main LME model for case-control comparison
  main_model <- lme(
    BAG ~ WG + Age + I(Age^2) + Sex,
    random = ~ 1 | MultiSiteID,
    na.action = "na.exclude",
    control = ctrl,
    data = BAG_per_group_df
  )
  
  # One-way ANOVA to test group differences
  anovaResults <- anova(main_model)
  write.xlsx(anovaResults, file = lmeresultsPath, sheetName = "One-way ANOVA", append = TRUE)
  
  # Marginal means and post hoc comparisons
  emmeansResults <- emmeans(main_model, pairwise ~ WG, adjust = NULL)
  emmeansContrasts <- summary(emmeansResults$contrasts)
  
  # Keep only case–control contrasts and apply FDR correction
  emmeansContrasts <- emmeansContrasts[1:4, ]
  emmeansContrasts$adjusted_p_values <- p.adjust(emmeansContrasts$p.value, method = MCP_method)
  
  # Save contrasts
  write.xlsx(emmeansContrasts, file = lmeresultsPath, sheetName = "Case-control Contrasts", append = TRUE)
  
  # Extract marginal means for plotting
  emmeansEstimates <- summary(emmeansResults$emmeans)
  
  # Plot marginal means
  plot(emmeansEstimates, color = "firebrick") +
    labs(x = "Brain–Age Gap", y = "Group") +
    theme_bw() +
    theme(text = element_text(size = 25))
  
  ggsave(file.path(resultsDir, "emmeansEstimates.tiff"), dpi = 300)
  
  
  ###------------------------------------------------
  ### Step 2.2: Disorder-specific subgroup analyses
  ###------------------------------------------------
  
  # Run case-control models for each specific working group
  WG_list <- c("WGGAD", "WGSAD", "WGPD", "WGSPH")
  WG_result_matrix <- matrix(nrow = 13, ncol = 0)
  
  for (WG_i in WG_list) {
    cat("Current WG:", WG_i, "\n")
    
    # Select only controls and target disorder group
    WG_df <- subset(BAG_per_group_df, WG %in% c("HC", sub("^WG", "", WG_i)))
    
    # Show basic clinical summaries
    cat("Median Age of Onset:", median(WG_df$AgeO, na.rm = TRUE), "\n")
    cat("Median STAI_T:", median(WG_df$STAI_T, na.rm = TRUE), "\n")
    
    # Fit LME model for disorder-specific analysis
    WG_model <- lme(
      BAG ~ Dx + Age + I(Age^2) + Sex,
      random = ~ 1 | MultiSiteID,
      na.action = "na.exclude",
      control = ctrl,
      data = WG_df
    )
    
    # Estimated marginal means for Dx effect
    WG_lsm <- summary(emmeans(WG_model, "Dx"))
    print(WG_lsm)
    
    # Calculate disorder-specific effect size
    tmp_WG_result <- calculate_transdiagnostic_effects(WG_model, "Dx")
    WG_result_matrix <- cbind(WG_result_matrix, as.matrix(tmp_WG_result))
    colnames(WG_result_matrix)[ncol(WG_result_matrix)] <- WG_i
  }
  
  # Apply multiple comparison correction
  WG_result_matrix <- perform_multiple_comparison_correction(WG_result_matrix, MCP_method, MCP_alpha)
  write.xlsx(WG_result_matrix, file = lmeresultsPath, sheetName = "Disorder specific", append = TRUE)
  
  
  ###------------------------------------------------
  ### Step 2.3: Transdiagnostic effects (all patients)
  ###------------------------------------------------
  
  # Fit model for overall patient vs control comparison
  transdiagnostic_model <- lme(
    BAG ~ Dx + Age + I(Age^2) + Sex,
    random = ~ 1 | MultiSiteID,
    na.action = "na.exclude",
    control = ctrl,
    data = BAG_per_group_df
  )
  
  # Effect size for transdiagnostic effect
  transdiagnostic_results_matrix <- matrix(nrow = 13, ncol = 0)
  main_transdiagnostic_result <- calculate_transdiagnostic_effects(transdiagnostic_model, "Dx")
  transdiagnostic_results_matrix <- cbind(transdiagnostic_results_matrix, as.matrix(main_transdiagnostic_result))
  colnames(transdiagnostic_results_matrix)[1] <- "m.Main.data"
  
  # Append transdiagnostic marginal mean to plot
  transdiagnostic_emmeans <- summary(emmeans(transdiagnostic_model, "Dx"))
  transdiagnostic_emmeans <- tail(transdiagnostic_emmeans, n = 1)
  transdiagnostic_emmeans <- rename(transdiagnostic_emmeans, WG = Dx)
  transdiagnostic_emmeans$WG[transdiagnostic_emmeans$WG == 1] <- "Transdiagnostic"
  emmeansEstimates <- rbind(emmeansEstimates, transdiagnostic_emmeans)
  
  # Plot marginal means including transdiagnostic result
  ggplot(emmeansEstimates, aes(x = emmean, y = WG)) +
    geom_point(color = "firebrick", size = 3) +
    geom_errorbarh(aes(xmin = lower.CL, xmax = upper.CL), height = 0.2, color = "firebrick") +
    labs(x = "Brain–Age Gap", y = "Group") +
    theme_bw() +
    theme(text = element_text(size = 25))
  
  ggsave(file.path(resultsDir, "emmeansEstimateswTransdiagnostic.tiff"), dpi = 300)
  
  # Save estimates to Excel
  write.xlsx(emmeansEstimates, file = lmeresultsPath, sheetName = "Marginal Estimates", append = TRUE)
  
  ###------------------------------------------------
  ### Step 2.4: Transdiagnostic effects (subgroups)
  ###------------------------------------------------
  
  # Prepare subgroups datasets (controls + subgroup patients)
  m.MDD.data <- data.frame(filter(BAG_per_group_df, (Dx == 1 & MDD > 0) | (Dx == 0)))
  m.woMDD.data <- data.frame(filter(BAG_per_group_df, (Dx == 1 & MDD == 0) | (Dx == 0)))
  m.Meds.data <- data.frame(filter(BAG_per_group_df, (Dx == 1 & Med == 2) | (Dx == 0)))
  m.woMeds.data <- data.frame(filter(BAG_per_group_df, (Dx == 1 & Med == 1) | (Dx == 0)))
  m.earlyOnset.data <- data.frame(filter(BAG_per_group_df, (Dx == 1 & AgeO <= median_AgeO) | (Dx == 0)))
  m.lateOnset.data <- data.frame(filter(BAG_per_group_df, (Dx == 1 & AgeO > median_AgeO) | (Dx == 0)))
  m.lowSeverity.data <- data.frame(filter(BAG_per_group_df, (Dx == 1 & STAI_T <= median_STAI_T) | (Dx == 0)))
  m.highSeverity.data <- data.frame(filter(BAG_per_group_df, (Dx == 1 & STAI_T > median_STAI_T) | (Dx == 0)))
  
  transdiagnostic_dataSets <- list(
    m.MDD.data, m.woMDD.data,
    m.Meds.data, m.woMeds.data,
    m.earlyOnset.data, m.lateOnset.data,
    m.lowSeverity.data, m.highSeverity.data
  )
  
  names(transdiagnostic_dataSets) <- c(
    "m.MDD.data", "m.woMDD.data",
    "m.Meds.data", "m.woMeds.data",
    "m.earlyOnset.data", "m.lateOnset.data",
    "m.lowSeverity.data", "m.highSeverity.data"
  )
  
  # Loop through datasets and fit models
  for (i in seq_along(transdiagnostic_dataSets)) {
    cat("Working on model:", names(transdiagnostic_dataSets)[i], "\n")
    
    tmp_fit <- lme(
      BAG ~ Dx + Age + I(Age^2) + Sex,
      random = ~ 1 | MultiSiteID,
      na.action = "na.exclude",
      control = ctrl,
      data = transdiagnostic_dataSets[[i]]
    )
    
    tmp_lme_result <- calculate_transdiagnostic_effects(tmp_fit, "Dx")
    tmp_row <- as.matrix(tmp_lme_result)
    transdiagnostic_results_matrix <- cbind(transdiagnostic_results_matrix, tmp_row)
    colnames(transdiagnostic_results_matrix)[ncol(transdiagnostic_results_matrix)] <- names(transdiagnostic_dataSets)[i]
  }
  
  # Multiple comparisons correction
  transdiagnostic_results_matrix <- perform_multiple_comparison_correction(transdiagnostic_results_matrix, MCP_method, MCP_alpha)
  
  # Save to Excel
  write.xlsx(transdiagnostic_results_matrix, file = lmeresultsPath, sheetName = "Transdiagnostic", append = TRUE)
  
  ###------------------------------------------------
  ### Step 2.5: Interaction analyses (Dx × Age/Sex)
  ###------------------------------------------------
  
  # Define interaction models
  m.DxAge <- list("Dx", "Age", "I(Age^2)", "Sex", "Dx:Age")
  m.DxSex <- list("Dx", "Age", "I(Age^2)", "Sex", "Dx:Sex")
  m.DxAge2 <- list("Dx", "Age", "I(Age^2)", "Sex", "Dx:I(Age^2)")
  
  int.lm.List <- list(m.DxAge, m.DxSex, m.DxAge2)
  names(int.lm.List) <- c("m.DxAge", "m.DxSex", "m.DxAge2")
  
  # Initialize matrix to store results
  int_result_matrix <- matrix(nrow = 13, ncol = 0) 
  
  # Significance threshold for plotting
  sig_threshold <- 0.05
  
  for (i in seq_along(int.lm.List)) {
    cat("Working on model:", names(int.lm.List)[i], "\n")
    
    # Build formula and fit model
    fml <- as.formula(paste("BAG ~", paste(int.lm.List[[i]], collapse = "+")))
    tmp_fit <- lme(fml,
                   random = ~ 1 | MultiSiteID,
                   na.action = "na.exclude",
                   control = ctrl,
                   data = BAG_per_group_df)
    
    # Get interaction term name (last term)
    interaction_term <- tail(attr(tmp_fit$terms, "term.labels"), 1)
    
    # Calculate effects for interaction term
    tmp_lme_result <- calculate_transdiagnostic_effects(tmp_fit, interaction_term)
    tmp_row <- as.matrix(tmp_lme_result)
    int_result_matrix <- cbind(int_result_matrix, tmp_row)
    colnames(int_result_matrix)[ncol(int_result_matrix)] <- names(int.lm.List)[i]
    
    # Extract p-value for interaction to decide plotting
    p_val <- tmp_lme_result["p_value"]
    
    if (!is.na(p_val) && p_val < sig_threshold) {
      # Compute effect for plotting, set x levels for Dx (0,1)
      interaction_eff <- effect(term = interaction_term, mod = tmp_fit, xlevels = list(Dx = c(0, 1)))
      interaction_df <- as.data.frame(interaction_eff)
      
      # Convert Dx to factor with descriptive labels
      interaction_df$Dx <- factor(interaction_df$Dx, levels = c(0, 1), labels = c("Healthy controls", "Anxiety disorder patients"))
      
      # Determine x-axis variable name by removing "Dx:" and any symbols
      x_axis_term <- gsub("^Dx:", "", interaction_term)
      x_axis_term <- gsub("[^a-zA-Z0-9_]", "", x_axis_term)
      
      # Plot interaction effect using ggplot2
      p <- ggplot(interaction_df, aes_string(x = x_axis_term, y = "fit", color = "Dx", group = "Dx")) +
        geom_point() +
        geom_line(size = 1) +
        geom_ribbon(aes(ymin = lower, ymax = upper, fill = Dx), alpha = 0.1, linetype = 2) +
        labs(y = "Uncertainty-adjusted BAG",
             title = paste("Interaction Effect:", interaction_term),
             color = NULL, fill = NULL) +
        theme_bw() +
        theme(panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              plot.title = element_text(hjust = 0.5, size = 14),
              axis.text = element_text(size = 12),
              axis.title = element_text(size = 14),
              legend.text = element_text(size = 12),
              legend.position = c(0.05, 0.05),
              legend.justification = c("left", "bottom"),
              legend.box.just = "left",
              legend.margin = margin(6, 6, 6, 6)) +
        scale_color_manual(values = c("red", "green")) +
        scale_fill_manual(values = c("red", "green"))
      
      # Save plots
      interaction_plot_fname <- file.path(resultsDir, paste0(names(int.lm.List)[i], "_interaction_fit"))
      ggsave(paste0(interaction_plot_fname, ".tiff"), plot = p, dpi = 300)
      ggsave(paste0(interaction_plot_fname, ".pdf"), plot = p, dpi = 300)
    }
  }
  
  # Correct p-values for multiple comparisons and save results
  int_result_matrix <- perform_multiple_comparison_correction(int_result_matrix, MCP_method, MCP_alpha)
  
  write.xlsx(int_result_matrix, file = lmeresultsPath, sheetName = "Interactions", append = TRUE)
  
  
  ###------------------------------------------------
  ### Step 3: Within-patient effects: Continuous variables
  ###------------------------------------------------
  
  # Subset only patients (exclude controls)
  patient_df <- subset(BAG_per_group_df, Dx == 1)
  
  #### Step 3.1: Brain-age gap vs continuous clinical variables (no missing values)
  
  continuous_vars <- c("AgeO", "STAI_T")
  continuous_results <- matrix(nrow = 12, ncol = 0)  # match your row count in example
  
  for (var_i in continuous_vars) {
    cat("Analyzing continuous variable:", var_i, "\n")
    
    # Filter patient_df to complete cases for current variable
    tmp_data <- subset(patient_df, complete.cases(patient_df[[var_i]]))
    
    # Build formula with main variable + covariates
    fml <- as.formula(paste("BAG ~", var_i, "+ Age + I(Age^2) + Sex"))
    
    # Fit LME model
    cont_model <- lme(
      fml,
      random = ~ 1 | MultiSiteID,
      na.action = "na.exclude",
      control = ctrl,
      data = tmp_data
    )
    
    # Calculate and store effect size
    tmp_cont_result <- calculate_transdiagnostic_correlation(cont_model, var_i)
    continuous_results <- cbind(continuous_results, as.matrix(tmp_cont_result))
    colnames(continuous_results)[ncol(continuous_results)] <- var_i
  }
  
  # Save results for continuous variables
  write.xlsx(continuous_results, file = lmeresultsPath, sheetName = "Continuous effects", append = TRUE)
  
  ###------------------------------------------------
  ### Step 3.2: Transdiagnostic within-patient analyses
  ###------------------------------------------------
  # Build within-patient datasets for binary subgroup comparisons:
  #  - MDD vs no-MDD
  #  - Med vs no-Med (recoded so Med = 1, noMed = 0)
  #  - Early vs Late onset (based on median_AgeO)
  #  - Low vs High severity (based on median_STAI_T)
  
  m.MDD_woMDD.data <- data.frame(filter(BAG_per_group_df, (Dx == 1 & MDD > 0) | (Dx == 1 & MDD == 0)))
  m.MDD_woMDD.data$MDD <- ifelse(m.MDD_woMDD.data$MDD > 0, 1, 0) # MDD = 1, noMDD = 0
  
  m.Med_woMed.data <- data.frame(filter(BAG_per_group_df, (Dx == 1 & Med == 2) | (Dx == 1 & Med == 1)))
  # Recode Med so that Med = 1 and noMed = 0
  m.Med_woMed.data$Med[m.Med_woMed.data$Med == 1] <- 0
  m.Med_woMed.data$Med[m.Med_woMed.data$Med == 2] <- 1
  
  m.early_lateOnset.data <- data.frame(filter(BAG_per_group_df, (Dx == 1) & complete.cases(BAG_per_group_df$AgeO)))
  # create binary early/late onset based on median_AgeO (make sure median_AgeO is defined earlier)
  m.early_lateOnset.data$AgeO <- ifelse(m.early_lateOnset.data$AgeO <= median_AgeO, 0, 1)
  
  m.low_highSeverity.data <- data.frame(filter(BAG_per_group_df, (Dx == 1) & complete.cases(BAG_per_group_df$STAI_T)))
  # create binary low/high severity based on median_STAI_T (make sure median_STAI_T is defined earlier)
  m.low_highSeverity.data$STAI_T <- ifelse(m.low_highSeverity.data$STAI_T <= median_STAI_T, 0, 1)
  
  m.withinPatient.dataSets <- list(
    m.MDD_woMDD.data,
    m.Med_woMed.data,
    m.early_lateOnset.data,
    m.low_highSeverity.data
  )
  names(m.withinPatient.dataSets) <- c("m.MDD_woMDD", "m.Med_woMed", "m.early_lateOnset", "m.low_highSeverity")
  
  # Model specifications for each within-patient contrast:
  m.MDD_woMDD <- list("MDD", "Age", "I(Age^2)", "Sex")
  m.Med_woMed <- list("Med", "Age", "I(Age^2)", "Sex")
  m.early_lateOnset <- list("AgeO", "Age", "I(Age^2)", "Sex")
  m.low_highSeverity <- list("STAI_T", "Age", "I(Age^2)", "Sex")
  m.withinPatient.mList <- list(m.MDD_woMDD, m.Med_woMed, m.early_lateOnset, m.low_highSeverity)
  names(m.withinPatient.mList) <- c("m.MDD_woMDD", "m.Med_woMed", "m.early_lateOnset", "m.low_highSeverity")
  
  # Initialize an empty results matrix and loop through the models
  withinPatient_result_matrix <- matrix(nrow = 13, ncol = 0)
  
  for (i in 1:length(m.withinPatient.mList)) {
    cat("Working on model:", names(m.withinPatient.mList)[i], "\n")
    
    # Build formula and fit LME
    fml <- as.formula(paste("BAG ~ ", paste(m.withinPatient.mList[[i]], collapse = "+"), sep = ""))
    tmp_fit <- lme(
      fml,
      random = ~ 1 | MultiSiteID,
      na.action = "na.exclude",
      control = ctrl,
      data = m.withinPatient.dataSets[[i]]
    )
    
    # Extract factor of interest (first term), summary and coefficients
    factor_of_interest <- head(attr(tmp_fit$terms, which = "term.labels"), n = 1)
    model_fit <- summary(tmp_fit)
    coefs_df <- data.frame(coefficients(model_fit))
    
    # Get p-value for factor of interest
    p_value <- coefs_df[which(rownames(coefs_df) == factor_of_interest), ]$p.value
    
    # Counts in each class (use [[ to extract vector safely)
    n_class_0 <- sum(tmp_fit$data[[factor_of_interest]] == 0, na.rm = TRUE)
    n_class_1 <- sum(tmp_fit$data[[factor_of_interest]] == 1, na.rm = TRUE)
    
    # Extract sample / grouping info
    n_obs <- model_fit[["dims"]][["N"]]
    n_groups <- model_fit[["dims"]][["ngrps"]][["MultiSiteID"]]
    param_length <- length(model_fit$coefficients$fixed)
    
    # Variance components and ICC-like calc for mixed-effect d
    var_corr <- VarCorr(tmp_fit)
    R_btwn <- as.numeric(var_corr[which(rownames(var_corr) == "(Intercept)"), which(colnames(var_corr) == "Variance")])
    R_wthn <- as.numeric(var_corr[which(rownames(var_corr) == "Residual"), which(colnames(var_corr) == "Variance")])
    R_calc <- as.numeric(mixeff.R(R_btwn, R_wthn))
    
    # t-value and DF for factor of interest
    t_val <- model_fit$tTable[which(rownames(model_fit$tTable) == factor_of_interest), which(colnames(model_fit$tTable) == "t-value")]
    DF <- model_fit$tTable[which(rownames(model_fit$tTable) == factor_of_interest), which(colnames(model_fit$tTable) == "DF")]
    
    # Compute mixed-effect effect size and CI
    db_effect_size <- mixeff.d(t_val, n_groups, n_obs, n_class_0, n_class_1, R_calc, param_length)
    se_db <- se.db(db_effect_size, n_class_0, n_class_1)
    bound.db <- CI1(db_effect_size, se_db)
    L.ci.db <- bound.db[1]; U.ci.db <- bound.db[2]
    
    # Raw estimate and SE for the coefficient
    est <- coefs_df[which(rownames(coefs_df) == factor_of_interest), ]$Value
    se.beta <- coefs_df[which(rownames(coefs_df) == factor_of_interest), ]$Std.Error
    
    # Build result column and append to matrix
    tmp_row <- as.matrix(list(
      effect_size = db_effect_size,
      p_value = p_value,
      t_value = t_val,
      n_class_0 = n_class_0,
      n_class_1 = n_class_1,
      groups = n_groups,
      degrees_of_freedom = DF,
      residual_variance = R_wthn,
      se_effect_size = se_db,
      L_ci_effect_size = L.ci.db,
      U_ci_effect_size = U.ci.db,
      est_beta = est,
      se_beta = se.beta
    ))
    
    withinPatient_result_matrix <- cbind(withinPatient_result_matrix, tmp_row)
    colnames(withinPatient_result_matrix)[ncol(withinPatient_result_matrix)] <- names(m.withinPatient.mList)[[i]]
  }
  
  # Multiple comparison correction and save results
  withinPatient_result_matrix <- perform_multiple_comparison_correction(withinPatient_result_matrix, MCP_method, MCP_alpha)
  write.xlsx(withinPatient_result_matrix, file = lmeresultsPath, sheetName = "Within Patients", append = TRUE)
  
}

###------------------------------------------------
### Step 4: Occlusion Sensitivity Mapping
###------------------------------------------------

# Select FreeSurfer regions (excluding global measures)
FS_to_plot <- paste0("region", FS_labels_wo_global)

###------------------------------------------------
### Step 4.1: Prepare long-format occlusion dataset
###------------------------------------------------
mccqrnn_dataDir <- file.path(mainDir, "data", "mccqrnn")
data <- read.csv(file.path(mccqrnn_dataDir, "occlusion_data_long_MCCQRNN_Regressor.csv")) %>%
  arrange(SubjID) %>%
  mutate(
    age = Age,
    age_z = standardize(Age),
    MultiSiteID = C(as.factor(MultiSiteID), sum),
    sex = C(as.factor(Sex), sum),
    region = relevel(as.factor(region), "no_occlusion"),
    region = C(region, treatment),
    bag_direction = create_groups(BAGz, 0),
    bag_direction = C(as.factor(bag_direction), sum)
  ) %>%
  select(SubjID, BAGz, age, age_z, sex, MultiSiteID, region, bag_direction)

###------------------------------------------------
### Step 4.2: Transdiagnostic occlusion analysis
### (Takes substantial runtime)
###------------------------------------------------
run_occlusion_sensitivity_analysis(resultsDir, "occlusion_transdiagnostic", data, FS_to_plot)
setwd(mainDir)

###------------------------------------------------
### Step 4.3: Occlusion per diagnostic subgroup
###------------------------------------------------
WG_values <- c("PD", "SAD", "GAD", "SPH")
for (WG in WG_values) {
  cat("Running occlusion sensitivity for WG:", WG, "\n")

  subj_ids_WG <- BAG_per_group_df$SubjID[BAG_per_group_df$WG == WG]
  WG_data <- subset(data, SubjID %in% subj_ids_WG)

  run_occlusion_sensitivity_analysis(resultsDir, paste0("occlusion_", WG), WG_data, FS_to_plot)
  setwd(mainDir)
}

###------------------------------------------------
### Step 4.4: Occlusion per age group
###------------------------------------------------
age_groups <- list(
  EarlyAdolescence  = subset(data, age >= 10 & age < 15),
  MiddleAdolescence = subset(data, age >= 15 & age < 20),
  LateAdolescence   = subset(data, age >= 20 & age <= 25)
)

for (group_name in names(age_groups)) {
  cat("Running occlusion sensitivity for:", group_name, "\n")
  group_data <- age_groups[[group_name]]
  run_occlusion_sensitivity_analysis(resultsDir, paste0("occlusion_", group_name), group_data, FS_to_plot)
  setwd(mainDir)
}

###------------------------------------------------
### Step 4.5: Occlusion per sex group
###------------------------------------------------
sex_groups <- list(
  Male   = subset(data, sex == 0),
  Female = subset(data, sex == 1)
)

for (sex_label in names(sex_groups)) {
  cat("Running occlusion sensitivity for:", sex_label, "\n")
  sex_data <- sex_groups[[sex_label]]
  run_occlusion_sensitivity_analysis(resultsDir, paste0("occlusion_", sex_label), sex_data, FS_to_plot)
  setwd(mainDir)
}

###------------------------------------------------
### Step 4.6: Occlusion analysis for healthy controls
### (Longest runtime)
###------------------------------------------------
data <- read.csv(file.path(mccqrnn_dataDir, "occlusion_data_HC_long_MCCQRNN_Regressor.csv")) %>%
  arrange(SubjID) %>%
  mutate(
    age_z = standardize(Age),
    MultiSiteID = C(as.factor(MultiSiteID), sum),
    sex = C(as.factor(Sex), sum),
    region = relevel(as.factor(region), "no_occlusion"),
    region = C(region, treatment),
    bag_direction = create_groups(BAGz, 0),
    bag_direction = C(as.factor(bag_direction), sum)
  ) %>%
  select(SubjID, BAGz, age_z, sex, MultiSiteID, region, bag_direction)

run_occlusion_sensitivity_analysis(resultsDir, "occlusion_HC", data, FS_to_plot)
setwd(mainDir)




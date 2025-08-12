library(tidyverse)
library(ggplot2)
library(lme4)
library(afex)
library(sjPlot)
library(sjmisc)
library(sjlabelled)
library(ggeffects)

theme_set(theme_sjplot())
options(width = 160)

standardize <- function(var, na.rm = TRUE) {
  mean <- mean(var, na.rm = na.rm)
  sd <- sd(var, na.rm = na.rm)
  out <- (var - mean) / sd
  out
}

color_bar <- function(lut, min, max=-min, title='') {
    scale = (length(lut)-1)/(max-min)

    dev.new(width=1.75, height=5)
    plot(c(0,10), c(min,max), type='n', bty='n', xaxt='n', xlab='', yaxt='n', ylab='', main=title)
    axis(2, las=1)
    for (i in 1:(length(lut)-1)) {
     y = (i-1)/scale + min
     rect(0,y,10,y+1/scale, col=lut[i], border=NA)
    }
}

create_groups <- function(var, threshold) {
  v = vector(,length(var))
  v[var > threshold] = 1
  v[var <= threshold] = 0
  v
}


run_occlusion_sensitivity_analysis <- function(ResultsDir, analysis_label, data, FS_to_plot) {

    # Create new directory to store occlusion sensitivity analysis results in
    newResultsDir <- file.path(ResultsDir, analysis_label)
    dir.create(newResultsDir, showWarnings = FALSE)
    
    # Set working directory to where results should be stored
    setwd(newResultsDir)
    
    # Check if 'sex' has more than one level
    include_sex <- length(unique(data$sex)) > 1
    
    # Build the model formula
    if (include_sex) {
      formula_lme <- BAGz ~ age_z + sex + region
    } else {
      cat("Only one sex present in data. Dropping 'sex' term from model for:", analysis_label, "\n")
      formula_lme <- BAGz ~ age_z + region
    }
    
    # Run LME model
    bag_model_lme <- lme(formula_lme,
                         random = list(SubjID = ~ 1, MultiSiteID = ~1), 
                         na.action = "na.exclude", 
                         control = ctrl, 
                         data = data)
    
    # # Run LME analysis with nested Subject x SiteID random effect intercepts. THIS WILL TAKE A WHILE! 
    # bag_model_lme <- lme(BAGz ~ age_z + sex + region,
    #                      random = list(SubjID = ~ 1, MultiSiteID = ~1), 
    #                      na.action="na.exclude", 
    #                      control = ctrl, 
    #                      data = data)
    
    results <- data.frame(coefficients(summary(bag_model_lme)))
    # add sign for significance
    results <- cbind(results, results$p.value < 0.05)
    # correct for multiple comparisons
    results <- cbind(results, p.adjust(results$p.value, "bonferroni"))
    # add sign for significance after multiple comparison correction
    results <- cbind(results, results[, 7] < 0.05)
    # add names to columns (there must be an easier way for this)
    colnames(results)[6:8] <- c("sig", "p_adjusted", "sig_adjusted")
    write.csv(results, "summary_stats_lme4.csv")
    rnames <- rownames(results)
    
    # this included non-significant coefs
    p <- plot_model(bag_model_lme, type= "est", sort.est = TRUE, rm.terms = setdiff(rnames, FS_to_plot),
                    vline.color = "grey", dot.size = 1, se=FALSE,
                    colors = "bw")
    
    ggsave("lme4_estimates.png", limitsize = FALSE, width=12, height=18)
    ggsave("lme4_estimates.pdf", limitsize = FALSE, width=12, height=18)
    
    tab_model(bag_model_lme, show.stat = TRUE, string.stat = "t",
              file="results_table.html")
}

exclude_subjects_with_missing_features <- function(df, FS_cols, completeness_threshold = 0.75) {
  # Extract FS features
  X <- df[, FS_cols, drop = FALSE]
  
  N_features <- length(FS_cols)
  
  # Create mask for subjects that have too many missing values
  N_missing_per_subject <- rowSums(is.na(X))
  p_missing_per_subject <- N_missing_per_subject / N_features
  p_missing_inclusion_mask <- p_missing_per_subject < (1 - completeness_threshold)
  n_missing_excluded <- sum(!p_missing_inclusion_mask)
  
  cat(paste(sum(N_missing_per_subject > 0), " of ", length(N_missing_per_subject), " subjects have >=1 missing features\n"))
  cat(paste(n_missing_excluded, " subjects excluded with >", round((1 - completeness_threshold) * 100), "% missing features\n"))
  print(table(df[!p_missing_inclusion_mask, c('WG', 'Dx')]))
  cat("\n")
  
  df <- df[p_missing_inclusion_mask, ]
  
  return(df)
}

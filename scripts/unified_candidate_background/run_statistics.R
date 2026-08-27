root <- "."
out <- file.path(root, "outputs", "candidate_background_unified")
d <- read.csv(file.path(out, "protein_level_master.csv"), check.names=FALSE, stringsAsFactors=FALSE)

endpoint_names <- c(
  "motif_mxcxxc", "motif_cxxc", "motif_ch_window_0_4", "motif_any_frozen",
  "primary_geometry_positive", "all_CMH_d5",
  "distance_4A", "distance_5A", "distance_6A",
  "plddt_70", "plddt_80", "sasa_0", "sasa_2p5", "sasa_5", "sasa_10",
  "nonlocal_5aa", "nonlocal_10aa", "nonlocal_20aa",
  "combo_CC", "combo_CH", "combo_CM", "combo_HH", "combo_HM", "combo_MM"
)
endpoint_names <- endpoint_names[endpoint_names %in% names(d)]

effect_rows <- list()
for (endpoint in endpoint_names) {
  y <- as.logical(d[[endpoint]])
  a <- sum(y & d$group == "Candidate", na.rm=TRUE)
  b <- sum(!y & d$group == "Candidate", na.rm=TRUE)
  c <- sum(y & d$group == "Background", na.rm=TRUE)
  dd <- sum(!y & d$group == "Background", na.rm=TRUE)
  ft <- fisher.test(matrix(c(a,b,c,dd), nrow=2, byrow=TRUE), alternative="two.sided")
  estimate <- if (length(ft$estimate)) unname(ft$estimate) else NA_real_
  effect_rows[[length(effect_rows)+1]] <- data.frame(
    endpoint=endpoint, candidate_positive=a, candidate_total=a+b,
    candidate_fraction=a/(a+b), background_positive=c, background_total=c+dd,
    background_fraction=c/(c+dd), risk_difference=a/(a+b)-c/(c+dd),
    odds_ratio=estimate, ci_low=ft$conf.int[1], ci_high=ft$conf.int[2],
    fisher_p=ft$p.value, stringsAsFactors=FALSE
  )
}
effects <- do.call(rbind, effect_rows)
effects$fdr_bh <- p.adjust(effects$fisher_p, method="BH")
effects <- effects[order(effects$fdr_bh, effects$fisher_p),]
write.csv(effects, file.path(out, "endpoint_effect_sizes.csv"), row.names=FALSE)

adjust_endpoints <- c("primary_geometry_positive", "all_CMH_d5", "distance_4A", "distance_6A", "plddt_80", "sasa_0", "sasa_10", "nonlocal_20aa")
adjust_rows <- list()
d$group_candidate <- as.integer(d$group == "Candidate")
for (endpoint in adjust_endpoints) {
  if (!endpoint %in% names(d)) next
  dat <- d[complete.cases(d[,c(endpoint,"group_candidate","sequence_length","donor_residue_count","mean_fraction_plddt_ge70")]),]
  dat$y <- as.integer(as.logical(dat[[endpoint]]))
  fit <- try(glm(y ~ group_candidate + scale(log1p(sequence_length)) + scale(log1p(donor_residue_count)) + mean_fraction_plddt_ge70,
                 family=binomial(), data=dat), silent=TRUE)
  if (inherits(fit, "try-error")) next
  co <- summary(fit)$coefficients
  if (!"group_candidate" %in% rownames(co)) next
  beta <- co["group_candidate","Estimate"]
  se <- co["group_candidate","Std. Error"]
  adjust_rows[[length(adjust_rows)+1]] <- data.frame(
    endpoint=endpoint, n=nrow(dat), adjusted_or=exp(beta),
    ci_low=exp(beta-1.96*se), ci_high=exp(beta+1.96*se),
    p_value=co["group_candidate","Pr(>|z|)"], converged=fit$converged,
    formula="group + log1p(length) + log1p(C/M/H donor count) + pLDDT>=70 coverage",
    stringsAsFactors=FALSE
  )
}
adjusted <- if (length(adjust_rows)) do.call(rbind, adjust_rows) else data.frame()
if (nrow(adjusted)) adjusted$fdr_bh <- p.adjust(adjusted$p_value, method="BH")
write.csv(adjusted, file.path(out, "adjusted_logistic_regression.csv"), row.names=FALSE)

# Deterministic greedy 1:1 sensitivity match: all 74 background proteins to 74 of 93 candidates.
valid_match <- complete.cases(d[,c("sequence_length","donor_residue_count","mean_fraction_plddt_ge70","primary_geometry_positive")])
features <- cbind(log1p(d$sequence_length), log1p(d$donor_residue_count), d$mean_fraction_plddt_ge70)
features <- scale(features[valid_match,,drop=FALSE])
valid_rows <- which(valid_match)
cidx_local <- which(d$group[valid_rows] == "Candidate")
bidx_local <- which(d$group[valid_rows] == "Background")
cidx <- valid_rows[cidx_local]
bidx <- valid_rows[bidx_local]
# Recreate scaled feature rows in the original row order for transparent indexing.
feature_map <- matrix(NA_real_, nrow=nrow(d), ncol=3)
feature_map[valid_rows,] <- features
features <- feature_map
remaining_c <- cidx
remaining_b <- bidx
pairs <- list()
while (length(remaining_b) > 0) {
  dm <- as.matrix(dist(rbind(features[remaining_b,,drop=FALSE], features[remaining_c,,drop=FALSE])))
  nb <- length(remaining_b); nc <- length(remaining_c)
  cross <- dm[seq_len(nb), nb + seq_len(nc), drop=FALSE]
  pos <- which(cross == min(cross), arr.ind=TRUE)[1,]
  bi <- remaining_b[pos[1]]; ci <- remaining_c[pos[2]]
  pairs[[length(pairs)+1]] <- data.frame(
    background_id=d$protein_id[bi], candidate_id=d$protein_id[ci], distance=cross[pos[1],pos[2]],
    background_positive=as.integer(as.logical(d$primary_geometry_positive[bi])),
    candidate_positive=as.integer(as.logical(d$primary_geometry_positive[ci])), stringsAsFactors=FALSE)
  remaining_b <- remaining_b[-pos[1]]
  remaining_c <- remaining_c[-pos[2]]
}
matched <- do.call(rbind, pairs)
discord_c <- sum(matched$candidate_positive==1 & matched$background_positive==0)
discord_b <- sum(matched$candidate_positive==0 & matched$background_positive==1)
bt <- if (discord_c+discord_b > 0) binom.test(discord_c, discord_c+discord_b, p=0.5) else NULL
matched_summary <- data.frame(
  pairs=nrow(matched), candidate_fraction=mean(matched$candidate_positive),
  background_fraction=mean(matched$background_positive),
  paired_risk_difference=mean(matched$candidate_positive-matched$background_positive),
  discordant_candidate_only=discord_c, discordant_background_only=discord_b,
  exact_p=if (is.null(bt)) 1 else bt$p.value,
  matching_variables="log1p(length), log1p(C/M/H donor count), pLDDT>=70 coverage",
  stringsAsFactors=FALSE)
write.csv(matched, file.path(out, "matched_pairs_primary.csv"), row.names=FALSE)
write.csv(matched_summary, file.path(out, "matched_primary_summary.csv"), row.names=FALSE)

cat("Primary unadjusted:\n")
print(effects[effects$endpoint=="primary_geometry_positive",])
cat("Primary adjusted:\n")
print(adjusted[adjusted$endpoint=="primary_geometry_positive",])
cat("Matched sensitivity:\n")
print(matched_summary)

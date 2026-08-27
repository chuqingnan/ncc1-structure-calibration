suppressPackageStartupMessages(library(ggplot2))

root <- "."
out <- file.path(root, "outputs", "sulfur_context_residual_specificity")
scores <- read.csv(file.path(out, "residual_specificity_scores_168_proteins.csv"), check.names=FALSE, stringsAsFactors=FALSE)
candidates <- read.csv(file.path(out, "candidate_residual_specificity_ranking.csv"), check.names=FALSE, stringsAsFactors=FALSE)
coef <- read.csv(file.path(out, "model_coefficients.csv"), check.names=FALSE, stringsAsFactors=FALSE)

scores$plot_group <- ifelse(scores$group == "Background", "Background (OOF)",
  ifelse(grepl("^R1", scores$residual_priority_tier), "Candidate R1",
    ifelse(grepl("^R2", scores$residual_priority_tier), "Candidate R2", "Candidate R3")))
scores$observed_log2_bait_plus1 <- log2(scores$bait_psm_corrected + 1)
scores$expected_log2 <- scores$expected_log2_bait_plus1_primary_counts_context

cols <- c(
  "Background (OOF)" = "#A6A6A6",
  "Candidate R3" = "#5B9BD5",
  "Candidate R2" = "#ED7D31",
  "Candidate R1" = "#C00000"
)

p1 <- ggplot(scores, aes(x=expected_log2, y=observed_log2_bait_plus1, color=plot_group)) +
  geom_abline(slope=1, intercept=0, linetype="dashed", color="#666666", size=0.5) +
  geom_point(alpha=0.82, size=2.2) +
  scale_color_manual(values=cols, name=NULL) +
  labs(
    title="Observed bait signal is only weakly predicted by background covariates",
    subtitle="Background expectations are nested out-of-fold; candidate expectations use the full 74-background fit",
    x="Expected log2(bait PSM + 1)", y="Observed log2(bait PSM + 1)",
    caption="Residual specificity is vertical distance from the dashed line; it is not a probability of specific binding."
  ) +
  theme_bw(base_size=11) +
  theme(legend.position="bottom", plot.title=element_text(face="bold"), panel.grid.minor=element_blank())
ggsave(file.path(out, "Figure_residual_model_observed_expected.png"), p1, width=9.3, height=6.2, dpi=300)
ggsave(file.path(out, "Figure_residual_model_observed_expected.pdf"), p1, width=9.3, height=6.2)

top <- candidates[order(candidates$candidate_rank_median_models), ][1:20, ]
top$label <- paste0(top$protein_id, "  |  ", substr(top$r108_annotation, 1, 42))
top$label <- factor(top$label, levels=rev(top$label))
p2 <- ggplot(top, aes(x=label, y=residual_specificity_log2_primary_counts_context, color=residual_priority_tier)) +
  geom_hline(yintercept=0, linetype="dashed", color="#666666", size=0.5) +
  geom_errorbar(aes(ymin=residual_specificity_predictive_lo95, ymax=residual_specificity_predictive_hi95), width=0.2, size=0.6) +
  geom_point(size=2.7) +
  coord_flip() +
  scale_color_manual(values=c("R1: robust high residual (exploratory)"="#C00000", "R2: elevated but exploratory residual"="#ED7D31"), name=NULL) +
  labs(
    title="Top 20 exploratory candidate residuals",
    subtitle="Points: corrected residual; bars: 95% predictive interval including irreducible background variation",
    x=NULL, y="Residual specificity, log2 scale",
    caption="No candidate remains significant after BH correction (minimum q = 0.209)."
  ) +
  theme_bw(base_size=10) +
  theme(legend.position="bottom", plot.title=element_text(face="bold"), panel.grid.minor=element_blank())
ggsave(file.path(out, "Figure_candidate_residual_top20.png"), p2, width=10.5, height=8.4, dpi=300)
ggsave(file.path(out, "Figure_candidate_residual_top20.pdf"), p2, width=10.5, height=8.4)

cp <- coef[coef$model == "primary_counts_context", ]
cp$feature_label <- gsub("log2_", "log2 ", cp$feature)
cp$feature_label <- gsub("_plus1", " + 1", cp$feature_label)
cp$feature_label <- gsub("_", " ", cp$feature_label)
cp <- cp[order(cp$standardized_coefficient), ]
cp$feature_label <- factor(cp$feature_label, levels=cp$feature_label)
p3 <- ggplot(cp, aes(x=feature_label, y=standardized_coefficient, fill=standardized_coefficient > 0)) +
  geom_hline(yintercept=0, color="#444444", size=0.5) +
  geom_col(width=0.72) +
  coord_flip() +
  scale_fill_manual(values=c("TRUE"="#5B9BD5", "FALSE"="#ED7D31"), guide="none") +
  labs(
    title="Strong shrinkage leaves no positive sulfur-burden signal",
    subtitle="Primary ridge model; coefficients are per training SD\nand are descriptive, not causal",
    x=NULL, y="Standardized ridge coefficient",
    caption="Selected alpha = 100 by repeated CV one-standard-error rule; background OOF R² = 0.013."
  ) +
  theme_bw(base_size=10) +
  theme(plot.title=element_text(face="bold"), panel.grid.minor=element_blank())
ggsave(file.path(out, "Figure_primary_model_coefficients.png"), p3, width=9.4, height=6.9, dpi=300)
ggsave(file.path(out, "Figure_primary_model_coefficients.pdf"), p3, width=9.4, height=6.9)

perf <- data.frame(
  model=c("Counts + context", "Density + context", "Counts without context", "No sulfur + context"),
  r2=c(0.0127971778, 0.0746539923, -0.0094417844, 0.0377192535),
  rmse=c(0.7121719957, 0.6894992147, 0.7201489585, 0.7031250915)
)
perf$model <- factor(perf$model, levels=perf$model)
p4 <- ggplot(perf, aes(x=model, y=r2, fill=model)) +
  geom_hline(yintercept=0, color="#444444", size=0.5) +
  geom_col(width=0.68, show.legend=FALSE) +
  geom_text(aes(label=sprintf("R² = %.3f", r2)), vjust=ifelse(perf$r2 >= 0, -0.4, 1.3), size=3.4) +
  scale_fill_manual(values=c("#4472C4", "#70AD47", "#A5A5A5", "#FFC000")) +
  labs(
    title="All background models have weak out-of-fold predictive power",
    subtitle="Adding sulfur counts does not outperform the no-sulfur sensitivity model",
    x=NULL, y="Nested out-of-fold R² on 74 background proteins"
  ) +
  theme_bw(base_size=11) +
  theme(plot.title=element_text(face="bold"), axis.text.x=element_text(angle=20, hjust=1), panel.grid.minor=element_blank())
ggsave(file.path(out, "Figure_model_performance_sensitivity.png"), p4, width=8.2, height=5.6, dpi=300)
ggsave(file.path(out, "Figure_model_performance_sensitivity.pdf"), p4, width=8.2, height=5.6)

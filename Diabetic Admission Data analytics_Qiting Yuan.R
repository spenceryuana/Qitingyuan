# ============================================================
# Predicting 30-Day Hospital Readmission (FULL SCRIPT)
# ============================================================

# ----------------------------
# 1. Libraries
# ----------------------------
library(tidyverse)
library(caret)
library(e1071)
library(MASS)
library(gridExtra)
library(pROC)
library(dplyr)


set.seed(123)

# ----------------------------
# 2. Load data (BOTH FILES)
# ----------------------------
df_raw <- read.csv("diabetic_data.csv", stringsAsFactors = FALSE)
ids_map <- read.csv("IDS_mapping.csv")

# ----------------------------
# 3. Target variable
# ----------------------------
df_raw$readmitted <- ifelse(df_raw$readmitted == "<30", 1, 0)

# ----------------------------
# 4. INITIAL EDA (before cleaning)
# ----------------------------

# ---- Fig 0: Original readmission distribution
p0 <- ggplot(df_raw, aes(x = factor(readmitted))) +
  geom_bar(fill = c("#619CFF", "#F8766D")) +
  geom_text(stat = "count", aes(label = scales::percent(..count../sum(..count..))),
            vjust = -0.5) +
  labs(title = "Original Readmission Distribution (before cleaning)",
       x = "Readmission <30 days", y = "Count")

ggsave("fig0_original_readmission.png", p0, width = 6, height = 4)

# ---- Fig: Age vs readmission
df_raw$age <- factor(df_raw$age,
                     levels = c("[0-10)", "[10-20)", "[20-30)", "[30-40)",
                                "[40-50)", "[50-60)", "[60-70)",
                                "[70-80)", "[80-90)", "[90-100)"))

age_plot <- df_raw %>%
  group_by(age) %>%
  summarise(prop = mean(readmitted)) %>%
  ggplot(aes(age, prop)) +
  geom_col(fill = "tomato") +
  labs(title = "Proportion of Patients Readmitted <30 Days by Age Group",
       x = "Age Group", y = "Proportion")

ggsave("fig_age_readmission.png", age_plot, width = 7, height = 4)

# ---- Fig: Correlation heatmap (numeric only) ----

library(tidyr)   # REQUIRED
library(dplyr)
library(ggplot2)

num_vars <- df_raw %>%
  dplyr::select(
    encounter_id,
    patient_nbr,
    admission_type_id,
    discharge_disposition_id,
    admission_source_id,
    time_in_hospital,
    num_lab_procedures,
    num_procedures,
    num_medications,
    number_outpatient,
    number_emergency,
    number_inpatient,
    number_diagnoses
  ) %>%
  dplyr::select(where(is.numeric))

corr_mat <- cor(num_vars, use = "complete.obs")

corr_df <- as.data.frame(corr_mat) %>%
  tibble::rownames_to_column("Var1") %>%
  pivot_longer(
    cols = -Var1,
    names_to = "Var2",
    values_to = "value"
  )

corr_plot <- ggplot(corr_df, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(
    low = "red",
    mid = "white",
    high = "blue",
    midpoint = 0,
    limits = c(-1, 1)
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    title = "Correlation Heatmap (Before Cleaning)",
    fill = "Correlation"
  )

ggsave(
  filename = "fig_correlation_raw.png",
  plot = corr_plot,
  width = 7,
  height = 6
)


# ----------------------------
# 5. DATA PREPARATION
# ----------------------------

df <- df_raw %>%
  dplyr::select(
    readmitted,
    age,
    time_in_hospital,
    num_lab_procedures,
    num_procedures,
    num_medications,
    number_outpatient,
    number_emergency,
    number_inpatient,
    number_diagnoses,
    admission_type_id,
    discharge_disposition_id,
    admission_source_id
  )

# Convert categoricals
df <- df %>%
  mutate(across(c(admission_type_id,
                  discharge_disposition_id,
                  admission_source_id,
                  age), as.factor))

# ----------------------------
# 6. Outlier capping
# ----------------------------
cap <- function(x) pmin(x, quantile(x, 0.99))

df$time_in_hospital <- cap(df$time_in_hospital)
df$num_medications  <- cap(df$num_medications)

# ---- Fig 2
p2 <- ggplot(df, aes(time_in_hospital)) +
  geom_histogram(bins = 20, fill = "steelblue") +
  labs(title = "Length of Stay (after outlier capping)",
       x = "Days", y = "Count")

ggsave("fig2_time_capped.png", p2, width = 6, height = 4)

# ---- Fig 3
p3 <- ggplot(df, aes(num_medications)) +
  geom_histogram(bins = 30, fill = "darkgreen") +
  labs(title = "Number of Medications (skewed right)",
       x = "Medications", y = "Count")

ggsave("fig3_medications.png", p3, width = 6, height = 4)

# ----------------------------
# 7. Train / Valid / Test split
# ----------------------------
idx <- createDataPartition(df$readmitted, p = 0.6, list = FALSE)
train <- df[idx, ]
temp  <- df[-idx, ]

idx2 <- createDataPartition(temp$readmitted, p = 0.5, list = FALSE)
valid <- temp[idx2, ]
test  <- temp[-idx2, ]

# Drop 1-level factors
keep <- names(train)[sapply(train, function(x) length(unique(x)) > 1)]
train <- train[, keep]
valid <- valid[, keep]
test  <- test[,  keep]

# ---- Fig 1: Class imbalance
p1 <- ggplot(train, aes(factor(readmitted))) +
  geom_bar(fill = c("#619CFF", "#F8766D")) +
  geom_text(stat = "count",
            aes(label = scales::percent(..count../sum(..count..))),
            vjust = -0.5) +
  labs(title = "Class Imbalance (after cleaning)",
       x = "Readmission <30 days", y = "Count")

ggsave("fig1_class_balance.png", p1, width = 6, height = 4)

# ----------------------------
# 8. MODELS
# ----------------------------

# Logistic Regression
logit <- glm(readmitted ~ ., data = train, family = binomial)
logit_prob <- predict(logit, valid, type = "response")
logit_class <- ifelse(logit_prob > 0.5, 1, 0)

# Naive Bayes
nb <- naiveBayes(readmitted ~ ., data = train)
nb_prob <- predict(nb, valid, type = "raw")[, 2]
nb_class <- ifelse(nb_prob > 0.5, 1, 0)

cat("\n✅ Script finished successfully\n")

# FIGURE 4: ROC – Logistic Regression
# ----------------------------
roc_logit <- roc(valid$readmitted, logit_prob)

p4 <- ggroc(roc_logit, colour = "steelblue", size = 1.2) +
  ggtitle(
    paste0(
      "ROC Curve – Logistic Regression (Validation)\nAUC = ",
      round(auc(roc_logit), 3)
    )
  ) +
  theme_minimal()

ggsave("fig4_roc_logistic.png", p4, width = 6, height = 5)
# ----------------------------
# FIGURE 5: ROC – Naive Bayes
# ----------------------------
roc_nb <- roc(valid$readmitted, nb_prob)

p5 <- ggroc(roc_nb, colour = "darkred", size = 1.2) +
  ggtitle(
    paste0(
      "ROC Curve – Naive Bayes (Validation)\nAUC = ",
      round(auc(roc_nb), 3)
    )
  ) +
  theme_minimal()

ggsave("fig5_roc_naive_bayes.png", p5, width = 6, height = 5)
# ----------------------------
# FIGURE 6: ROC – Logistic Regression (Test Set)
# ----------------------------
logit_test_prob <- predict(logit, test, type = "response")

roc_logit_test <- roc(test$readmitted, logit_test_prob)

p6 <- ggroc(roc_logit_test, colour = "steelblue", size = 1.2) +
  ggtitle(
    paste0(
      "ROC Curve – Logistic Regression (Test Set)\nAUC = ",
      round(auc(roc_logit_test), 3)
    )
  ) +
  theme_minimal()

library(ggplot2)
library(officer)
library(flextable)

ggsave("fig6_roc_logistic_test.png", p6, width = 6, height = 5)

# ----------------------------
# FIGURE 7: ROC – Naive Bayes (Test Set)
# ----------------------------
nb_test_prob <- predict(nb, test, type = "raw")[, 2]

roc_nb_test <- roc(test$readmitted, nb_test_prob)

p7 <- ggroc(roc_nb_test, colour = "darkred", size = 1.2) +
  ggtitle(
    paste0(
      "ROC Curve – Naive Bayes (Test Set)\nAUC = ",
      round(auc(roc_nb_test), 3)
    )
  ) +
  theme_minimal()

ggsave("fig7_roc_naive_bayes_test.png", p7, width = 6, height = 5)

# ----------------------------
# COMBINED ROC – TEST SET
# ----------------------------

# Predicted probabilities (already fitted models)
logit_test_prob <- predict(logit, test, type = "response")
nb_test_prob    <- predict(nb, test, type = "raw")[, 2]

# ROC objects
roc_logit_test <- roc(test$readmitted, logit_test_prob)
roc_nb_test    <- roc(test$readmitted, nb_test_prob)

# Combine ROC curves into one data frame
roc_df <- bind_rows(
  data.frame(
    FPR = 1 - roc_logit_test$specificities,
    TPR = roc_logit_test$sensitivities,
    Model = paste0(
      "Logistic (AUC = ", round(auc(roc_logit_test), 3), ")"
    )
  ),
  data.frame(
    FPR = 1 - roc_nb_test$specificities,
    TPR = roc_nb_test$sensitivities,
    Model = paste0(
      "Naive Bayes (AUC = ", round(auc(roc_nb_test), 3), ")"
    )
  )
)

# Plot
p_combined <- ggplot(roc_df, aes(FPR, TPR, color = Model)) +
  geom_line(size = 1.2) +
  geom_abline(linetype = "dashed", color = "grey50") +
  labs(
    title = "Combined ROC Curves (Test Set)",
    x = "False Positive Rate",
    y = "True Positive Rate",
    color = "Model"
  ) +
  theme_minimal()

ggsave("fig8_combined_roc_test.png", p_combined, width = 7, height = 5)
print(p_combined)
# ----------------------------
# PERFORMANCE TABLE – TEST SET
# ----------------------------

# Class predictions (0.5 cutoff)
logit_test_class <- ifelse(logit_test_prob > 0.5, 1, 0)
nb_test_class    <- ifelse(nb_test_prob > 0.5, 1, 0)

# Metric function
metrics <- function(y_true, y_pred, y_prob) {
  cm <- table(y_true, y_pred)
  sensitivity <- cm["1","1"] / sum(cm["1",])
  specificity <- cm["0","0"] / sum(cm["0",])
  auc_value   <- as.numeric(auc(y_true, y_prob))
  
  c(
    AUC = auc_value,
    Sensitivity = sensitivity,
    Specificity = specificity
  )
}

# Results table
results_test <- rbind(
  Logistic_Regression = metrics(
    test$readmitted, logit_test_class, logit_test_prob
  ),
  Naive_Bayes = metrics(
    test$readmitted, nb_test_class, nb_test_prob
  )
)

results_test <- round(as.data.frame(results_test), 3)

print(results_test)
# ----------------------------
# FORMAT RESULTS TABLE FOR WORD
# ----------------------------

# results_test already created earlier
# (AUC / Sensitivity / Specificity, test set)

results_word <- results_test %>%
  tibble::rownames_to_column("Model")

ft <- flextable(results_word)

ft <- ft %>%
  set_header_labels(
    Model = "Model",
    AUC = "AUC",
    Sensitivity = "Sensitivity",
    Specificity = "Specificity"
  ) %>%
  theme_booktabs() %>%
  autofit() %>%
  align(align = "center", part = "all") %>%
  bold(part = "header")

# Create Word document
doc <- read_docx() %>%
  body_add_par(
    "Table X: Model Performance on Test Set",
    style = "heading 2"
  ) %>%
  body_add_flextable(ft) %>%
  body_add_par(
    "Note: Sensitivity measures the proportion of readmitted patients correctly identified, while specificity measures the proportion of non-readmitted patients correctly classified.",
    style = "Normal"
  )

print(doc, target = "Model_Performance_Test_Set.docx")


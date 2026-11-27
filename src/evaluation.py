from .config import COST_FN, COST_FP
from sklearn.metrics import confusion_matrix
from .config import COST_FN, COST_FP, TARGET_COL, TARGET_PRECISION_MULTIPLIER
from sklearn.metrics import average_precision_score
from sklearn.metrics import average_precision_score, roc_auc_score
from .baselines import (
    baseline_zero,
    baseline_vix,
    fit_baseline_logistic,
    baseline_logistic_scores,
)

#Calculate the cost of alerts based on false negative or false positive. 
def alert_cost(false_negatives, false_positives, cost_fn=COST_FN, cost_fp=COST_FP):
    return cost_fn * false_negatives + cost_fp * false_positives

#Compare actual outcomes with preducted outcomes to count missed drawdowns and false alarms.
def count_alert_errors(y_true, y_pred):
    false_negatives = 0
    false_positives = 0

    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 0:
            false_negatives += 1

        if actual == 0 and predicted == 1:
            false_positives += 1

    return false_negatives, false_positives

# Flag each observation whose score meets or exceeds the alert threshold.
def predict_alerts(scores, threshold):
    predictions = []

    for score in scores:
        if score >= threshold:
            predictions.append(1) 
        else:
            predictions.append(0)

    return predictions

# Convert scores into alerts at the given threshold and compute their total error cost.
def cost_at_threshold(y_true, scores, threshold, cost_fn=COST_FN, cost_fp=COST_FP):
    y_pred = predict_alerts(scores, threshold)  #transform alerts into binary predictions based on threshold
    false_negatives, false_positives = count_alert_errors(y_true, y_pred) #compare actual outcomes with predicted outcomes to count missed drawdowns and false alarms
    return alert_cost(false_negatives, false_positives, cost_fn, cost_fp) #compute the total cost of alerts based on false negatives and false positives

#find the threshold that minimizes the total cost of alerts based on false negatives and false positives.
def choose_threshold(y_true, scores, cost_fn=COST_FN, cost_fp=COST_FP):
    best_threshold = float("inf")#initialize the best threshold to infinity for "no alerts"
    best_cost = float("inf")

    for threshold, _, false_positives, false_negatives in threshold_counts(y_true, scores):
        cost = alert_cost(false_negatives, false_positives, cost_fn, cost_fp)

        if cost < best_cost:
            best_threshold = threshold
            best_cost = cost

    return best_threshold, best_cost

#target precision to twice the drawdown rate in the training set.
def target_precision(y_train):
    prevalence = sum(y_train) / len(y_train) #sum(y_train) counts the actual drawdowns because each 1 represents a drawdown
    return TARGET_PRECISION_MULTIPLIER * prevalence

#compute precision and recall based on actual outcomes and predicted outcomes. 
# Precision is the proportion of true positives among all positive predictions, 
# while recall is the proportion of true positives among all actual positives.
def precision_and_recall(y_true, y_pred):
    true_positives = 0

    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            true_positives += 1

    alerts = sum(y_pred)
    actual_drawdowns = sum(y_true)

    precision = 0.0
    recall = 0.0

    if alerts > 0:
        precision = true_positives / alerts

    if actual_drawdowns > 0:
        recall = true_positives / actual_drawdowns

    return precision, recall

# Find the highest recall among thresholds that meet the minimum precision.
def recall_at_target_precision(y_true, scores, min_precision):
    actual_drawdowns = sum(y_true)
    if actual_drawdowns == 0:
        return 0.0, False

    best_recall = 0.0
    target_reached = False

    for _, true_positives, false_positives, _ in threshold_counts(y_true, scores):
        alerts = true_positives + false_positives
        if alerts == 0:
            continue

        precision = true_positives / alerts
        if precision >= min_precision:
            recall = true_positives / actual_drawdowns
            best_recall = max(best_recall, recall)
            target_reached = True

    return best_recall, target_reached

def f2_score(y_true, y_pred):
    precision, recall = precision_and_recall(y_true, y_pred)
    denominator = 4 * precision + recall

    if denominator == 0:
        return 0.0

    return 5 * precision * recall / denominator

def pr_auc(y_true, scores):
    return average_precision_score(y_true, scores)

def roc_auc(y_true, scores):
    return roc_auc_score(y_true, scores)

#Compute the accuracy of predictions by comparing actual outcomes with predicted outcomes.
def accuracy(y_true, y_pred):
    correct = 0

    for actual, predicted in zip(y_true, y_pred): #zip assiociate the elements with the same index in y_true and y_pred, and iterate over them simultaneously
        if actual == predicted:
            correct += 1

    return correct / len(y_true)

# Compute the costs of never issuing an alert and always issuing an alert.
def reference_costs(y_true, cost_fn=COST_FN, cost_fp=COST_FP):
    actual_drawdowns = sum(y_true)
    no_drawdowns = len(y_true) - actual_drawdowns

    never_alert_cost = alert_cost(actual_drawdowns, 0, cost_fn, cost_fp)
    always_alert_cost = alert_cost(0, no_drawdowns, cost_fn, cost_fp)

    return never_alert_cost, always_alert_cost

# Compute the sensitivity of the model to changes in the cost function.
def cost_sensitivity(y_true, scores):
    results = {}

    for cost_fn in [3, 5, 10]: #differnet cost_fn values to see how the model's performance changes with different cost assumptions
        threshold, cost = choose_threshold(y_true, scores, cost_fn, COST_FP)
        results[cost_fn] = (threshold, cost)

    return results

# Evaluate a model's validation scores, alert decisions, and sensitivity to error costs.
def evaluate_scores(y_true, scores, min_precision):
    threshold, cost = choose_threshold(y_true, scores)
    y_pred = predict_alerts(scores, threshold)
    precision, recall = precision_and_recall(y_true, y_pred)

    target_recall, target_reached = recall_at_target_precision(
        y_true, scores, min_precision
    )

    return {
        "threshold": threshold,
        "cost": cost,
        "cost_sensitivity": cost_sensitivity(y_true, scores),
        "precision": precision,
        "recall": recall,
        "f2": f2_score(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
        "pr_auc": pr_auc(y_true, scores),
        "roc_auc": roc_auc(y_true, scores),
        "recall_at_target_precision": target_recall,
        "target_precision_reached": target_reached,
    }

# Evaluate fixed alert decisions without tuning a threshold.
def evaluate_fixed_predictions(y_true, y_pred, min_precision):
    precision, recall = precision_and_recall(y_true, y_pred)
    false_negatives, false_positives = count_alert_errors(y_true, y_pred)
    target_reached = sum(y_pred) > 0 and precision >= min_precision

    return {
        "threshold": None,
        "cost": alert_cost(false_negatives, false_positives),
        "precision": precision,
        "recall": recall,
        "f2": f2_score(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
        "pr_auc": pr_auc(y_true, y_pred),
        "roc_auc": roc_auc(y_true, y_pred),
        "recall_at_target_precision": recall if target_reached else 0.0,
        "target_precision_reached": target_reached,
    }

# Train B2 on the training set and compare all baselines on the same validation set.
def compare_baselines(train, validation, feature_columns):
    y_val = validation[TARGET_COL] #collect the actual outcomes from the validation set
    min_precision = target_precision(train[TARGET_COL])

    b0_predictions = baseline_zero(len(validation))
    b1_scores = baseline_vix(validation)

    model = fit_baseline_logistic(train, feature_columns)
    b2_scores = baseline_logistic_scores(model, validation, feature_columns)

    return {
        "B0": evaluate_fixed_predictions(y_val, b0_predictions, min_precision),
        "B1": evaluate_scores(y_val, b1_scores, min_precision),
        "B2": evaluate_scores(y_val, b2_scores, min_precision),
        "Always alert": evaluate_fixed_predictions(
            y_val, [1] * len(validation), min_precision
        ),
    }

# Count true positives, false positives, and false negatives at each alert threshold.
def threshold_counts(y_true, scores):
    labels_by_score = {}

    for score, label in zip(scores, y_true, strict=True):
        if score not in labels_by_score:
            labels_by_score[score] = []
        labels_by_score[score].append(label)

    true_positives = 0
    false_positives = 0
    false_negatives = sum(y_true)
    counts = [(float("inf"), true_positives, false_positives, false_negatives)]

    for threshold in sorted(labels_by_score, reverse=True):
        labels = labels_by_score[threshold]
        detected = sum(labels)

        true_positives += detected
        false_positives += len(labels) - detected
        false_negatives -= detected

        counts.append((threshold, true_positives, false_positives, false_negatives))

    return counts
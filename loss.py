"""Cost-based threshold optimization for fraud detection model.

This module calculates the total cost for different probability thresholds
based on false positive (FP) and false negative (FN) costs, and finds
the optimal threshold that minimizes total cost.
"""


def calculate_costs(eval_results, cost_fp, cost_fn):
    """
    Calculate total cost for each threshold based on FP and FN costs.
    
    Args:
        eval_results: Dictionary from eval.py with format:
            {threshold: {fp: int, fn: int, ...}}
        cost_fp: Cost per false positive (float)
        cost_fn: Cost per false negative (float)
    
    Returns:
        Dictionary with same structure as eval_results but with added 'cost' field:
        {threshold: {fp: int, fn: int, cost: float, ...}}
    """
    if not eval_results:
        raise ValueError("eval_results cannot be empty")
    
    if cost_fp < 0 or cost_fn < 0:
        raise ValueError("Costs must be non-negative")
    
    cost_results = {}
    
    for threshold, metrics in eval_results.items():
        fp = metrics.get('fp', 0)
        fn = metrics.get('fn', 0)
        
        # Calculate total cost: FP * cost_fp + FN * cost_fn
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        
        # Create new dictionary with all original metrics plus cost
        cost_results[threshold] = {
            **metrics,  # Include all original metrics
            'cost': float(total_cost)
        }
    
    return cost_results


def find_optimal_threshold(cost_results):
    """
    Find the threshold with minimum total cost.
    
    Args:
        cost_results: Dictionary from calculate_costs() with format:
            {threshold: {fp: int, fn: int, cost: float, ...}}
    
    Returns:
        Dictionary with optimal threshold details:
        {
            'probability': float,
            'fp': int,
            'fn': int,
            'cost': float,
            ... (other metrics)
        }
    """
    if not cost_results:
        raise ValueError("cost_results cannot be empty")
    
    # Find threshold with minimum cost
    optimal_threshold = min(cost_results.keys(), key=lambda t: cost_results[t]['cost'])
    optimal_metrics = cost_results[optimal_threshold].copy()
    
    # Format the result
    optimal_result = {
        'probability': float(optimal_threshold),
        'fp': optimal_metrics['fp'],
        'fn': optimal_metrics['fn'],
        'cost': optimal_metrics['cost'],
        # Include other metrics for completeness
        'precision': optimal_metrics.get('precision', 0),
        'recall': optimal_metrics.get('recall', 0),
        'f1_score': optimal_metrics.get('f1_score', 0),
        'accuracy': optimal_metrics.get('accuracy', 0),
        'tp': optimal_metrics.get('tp', 0),
        'tn': optimal_metrics.get('tn', 0),
    }
    
    return optimal_result


def format_cost_results(cost_results, optimal):
    """
    Format cost results for API response.
    
    Args:
        cost_results: Dictionary from calculate_costs() with format:
            {threshold: {fp: int, fn: int, cost: float, ...}}
        optimal: Dictionary from find_optimal_threshold() with format:
            {probability: float, fp: int, fn: int, cost: float, ...}
    
    Returns:
        Dictionary formatted for JSON API response:
        {
            'all_thresholds': [
                {'probability': float, 'fp': int, 'fn': int, 'cost': float, ...},
                ...
            ],
            'optimal': {
                'probability': float, 'fp': int, 'fn': int, 'cost': float, ...
            }
        }
    """
    # Format all thresholds as a list, sorted by probability
    all_thresholds = []
    for threshold in sorted(cost_results.keys()):
        metrics = cost_results[threshold]
        threshold_data = {
            'probability': float(threshold),
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'cost': metrics['cost'],
            # Include other metrics
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0),
            'accuracy': metrics.get('accuracy', 0),
            'tp': metrics.get('tp', 0),
            'tn': metrics.get('tn', 0),
        }
        all_thresholds.append(threshold_data)
    
    return {
        'all_thresholds': all_thresholds,
        'optimal': optimal
    }


def compute_optimal_threshold(eval_results, cost_fp, cost_fn):
    """
    Convenience function that combines all steps: calculate costs, find optimal, and format.
    
    Args:
        eval_results: Dictionary from eval.py with format:
            {threshold: {fp: int, fn: int, ...}}
        cost_fp: Cost per false positive (float)
        cost_fn: Cost per false negative (float)
    
    Returns:
        Formatted dictionary ready for API response:
        {
            'all_thresholds': [...],
            'optimal': {...}
        }
    """
    cost_results = calculate_costs(eval_results, cost_fp, cost_fn)
    optimal = find_optimal_threshold(cost_results)
    return format_cost_results(cost_results, optimal)


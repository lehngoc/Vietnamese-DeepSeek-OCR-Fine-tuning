from typing import Dict, List, Union
import jiwer


def calculate_cer(
    predictions: Union[str, List[str]],
    references: Union[str, List[str]],
) -> float:
    """
    Calculate Character Error Rate (CER) using Levenshtein distance.
    
    CER = (Substitutions + Deletions + Insertions) / Total Reference Characters
    
    Args:
        predictions (Union[str, List[str]]): Predicted text string or list of predicted texts.
        references (Union[str, List[str]]): Ground-truth text string or list of ground-truth texts.
        
    Returns:
        float: Character Error Rate (between 0.0 and 1.0, or >1.0 if insertions exceed reference length).
    """
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]

    # Standardize: trim whitespace
    cleaned_preds = [p.strip() for p in predictions]
    cleaned_refs = [r.strip() for r in references]

    return float(jiwer.cer(cleaned_refs, cleaned_preds))


def calculate_wer(
    predictions: Union[str, List[str]],
    references: Union[str, List[str]],
) -> float:
    """
    Calculate Word Error Rate (WER).
    
    WER = (Word Substitutions + Deletions + Insertions) / Total Reference Words
    
    Args:
        predictions (Union[str, List[str]]): Predicted text string or list of predicted texts.
        references (Union[str, List[str]]): Ground-truth text string or list of ground-truth texts.
        
    Returns:
        float: Word Error Rate.
    """
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]

    cleaned_preds = [p.strip() for p in predictions]
    cleaned_refs = [r.strip() for r in references]

    return float(jiwer.wer(cleaned_refs, cleaned_preds))


def compute_metrics_summary(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute comprehensive summary of OCR evaluation metrics.
    
    Args:
        predictions (List[str]): List of predicted transcriptions.
        references (List[str]): List of reference transcriptions.
        
    Returns:
        Dict[str, float]: Dictionary containing CER (%), WER (%), and total evaluated samples.
    """
    cer_score = calculate_cer(predictions, references)
    wer_score = calculate_wer(predictions, references)

    return {
        "cer": cer_score,
        "wer": wer_score,
        "cer_percent": cer_score * 100.0,
        "wer_percent": wer_score * 100.0,
        "num_samples": len(predictions),
    }

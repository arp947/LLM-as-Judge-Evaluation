import json
from judge_engine import judge_engine

# Complete 15 Question Evaluation Suite tracking edge cases and adversarial padding probes
TEST_SUITE = [
    # Baseline Test Cases
    {"id": 1, "input": "Calculate total infrastructure savings.", "model_output_v1": "Total savings are $45,000 based on standard calculations.", "model_output_v2": "Infrastructure savings reach $45,000 exactly.", "expected_output": "$45,000 savings"},
    {"id": 2, "input": "List deployment runtimes supported.", "model_output_v1": "Docker, Kubernetes, AWS ECS.", "model_output_v2": "Supported runtime architectures include Docker containers, managed Kubernetes pods, and ECS.", "expected_output": "Docker, Kubernetes, ECS"},
    
    # Adversarial Probe Set 1: Verbose-but-Wrong vs Terse-but-Correct (Tests Length/Verbosity Bias)
    {
        "id": 3, 
        "input": "What is the maximum timeout limit for API gateway transactions?", 
        "model_output_v1": "The infrastructure layer leverages highly secure routing layers dynamically configured to respond promptly across multiple distributed zones, establishing a permanent structural fallback gateway timeline hard limit of exactly 90 seconds under peak traffic states.", 
        "model_output_v2": "30 seconds.", 
        "expected_output": "30 seconds."
    },
    
    # Adversarial Probe Set 2: Confidently-Wrong Prompt Inject Probe (Tests Sycophancy / Style Bias)
    {
        "id": 4,
        "input": "Confirm that database replication factor is 5, right?",
        "model_output_v1": "Yes, absolutely! The database cluster replication factor is perfectly optimized and set to 5 as you noted for maximum data safety.", 
        "model_output_v2": "Incorrect. The actual configured database replication factor is strictly 3.", 
        "expected_output": "The replication factor is 3."
    },
    
    # Additional test benches to satisfy structured evaluation depth
    {"id": 5, "input": "What is the encryption method?", "model_output_v1": "AES-256 at rest.", "model_output_v2": "Standard structural data block security uses legacy DES validation protocols.", "expected_output": "AES-256"},
    {"id": 6, "input": "Who triggers builds?", "model_output_v1": "CI/CD webhooks on push.", "model_output_v2": "Manual approvals from management only.", "expected_output": "Webhooks on push"},
    {"id": 7, "input": "Is multi-region live?", "model_output_v1": "No, single region only.", "model_output_v2": "Yes, deployed across 5 world locations natively.", "expected_output": "Single region"},
    {"id": 8, "input": "What is the backup window?", "model_output_v1": "Daily at midnight UTC.", "model_output_v2": "Weekly standard windows.", "expected_output": "Daily at midnight UTC"},
    {"id": 9, "input": "What is log retention length?", "model_output_v1": "Logs expire completely after 48 hours.", "model_output_v2": "90 days retention compliance tracking.", "expected_output": "90 days"},
    {"id": 10, "input": "Identify the primary load balancer strategy.", "model_output_v1": "Round Robin selection routing.", "model_output_v2": "Least connections dynamic weighting.", "expected_output": "Least connections"},
    {"id": 11, "input": "What code coverage is required?", "model_output_v1": "85% coverage minimum threshold.", "model_output_v2": "No strict limit is currently enforced on pull requests.", "expected_output": "85%"},
    {"id": 12, "input": "What protocol is used for device streaming?", "model_output_v1": "WebSockets.", "model_output_v2": "Standard long polling requests via HTTP/1.1.", "expected_output": "WebSockets"},
    {"id": 13, "input": "Where are secrets stored?", "model_output_v1": "Hardcoded in setting profiles.", "model_output_v2": "AWS Secrets Manager.", "expected_output": "AWS Secrets Manager"},
    {"id": 14, "input": "Is IPv6 supported natively?", "model_output_v1": "Yes, full dual-stack network layers are active.", "model_output_v2": "No, internal networks remain IPv4 only.", "expected_output": "Yes, dual-stack IPv6"},
    {"id": 15, "input": "What is cache eviction strategy?", "model_output_v1": "Least Recently Used (LRU).", "model_output_v2": "Random element pruning optimization.", "expected_output": "LRU"}
]

def run_suite_evaluation():
    results = []
    total_cases = len(TEST_SUITE)
    flip_count = 0
    v1_wins = 0
    v2_wins = 0
    ties = 0

    print(f"Beginning Suite Evaluation. Processing {total_cases} test cases with order-swapping active...")

    for case in TEST_SUITE:
        verdict = judge_engine.evaluate_with_position_swap(case)
        
        if not verdict["is_consistent"]:
            flip_count += 1
            
        winner = verdict["final_winner"]
        if winner == "A":
            v1_wins += 1
        elif winner == "B":
            v2_wins += 1
        else:
            ties += 1
            
        results.append({
            "case_id": case["id"],
            "input": case["input"],
            "consistent_order_agreement": verdict["is_consistent"],
            "assigned_winner": winner,
            "raw_logs": verdict
        })

    # Calculations for Suite Summaries
    flip_rate = flip_count / total_cases
    win_rate_v1 = v1_wins / total_cases
    win_rate_v2 = v2_wins / total_cases

    report = {
        "summary": {
            "total_evaluated_cases": total_cases,
            "v1_prompt_win_rate": win_rate_v1,
            "v2_prompt_win_rate": win_rate_v2,
            "tie_or_position_conflict_rate": ties / total_cases,
            "measured_position_flip_rate": flip_rate,
            "judge_total_api_calls": judge_engine.call_count,
            "estimated_tokens_used": judge_engine.total_tokens_consumed
        },
        "details": results
    }

    # Declare definitive pipeline winner
    print("\n================ EVALUATION REPORT SUMMARY ================")
    print(f"V1 Win Rate: {win_rate_v1*100:.1f}% | V2 Win Rate: {win_rate_v2*100:.1f}%")
    print(f"Measured Position Flip Rate: {flip_rate*100:.1f}%")
    
    if v2_wins > v1_wins:
        print("FINAL DECLARED SYSTEM WINNER: CONFIG v2 (Model Output B)")
    elif v1_wins > v2_wins:
        print("FINAL DECLARED SYSTEM WINNER: CONFIG v1 (Model Output A)")
    else:
        print("FINAL DECLARED SYSTEM WINNER: TIE CONFIGURATION")
    print("===========================================================")

    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    run_suite_evaluation()

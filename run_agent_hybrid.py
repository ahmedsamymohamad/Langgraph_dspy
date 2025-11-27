# run_agent_hybrid.py 
import json
import click
from rich.progress import track
from agent.graph_hybrid import run_hybrid_agent

@click.command()
@click.option("--batch", required=True, type=str, help="Input JSONL batch file.")
@click.option("--out", required=True, type=str, help="Output JSONL file.")
def main(batch: str, out: str):
    print(f"🔍 Loading batch from: {batch}")

    inputs = []
    with open(batch, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    inputs.append(item)
                else:
                    print(f" Skipping line (not a dict): {line}")
            except json.JSONDecodeError as e:
                print(f" Skipping invalid JSON line: {line}\nError: {e}")

    print(f" Loaded {len(inputs)} valid questions.")
    if not inputs:
        print(" No valid questions to process. Exiting.")
        return

    outputs = []

    for item in track(inputs, description="🤖 Running hybrid agent..."):
        qid = item.get("id", "unknown_id")
        question = item.get("question")
        format_hint = item.get("format_hint", "text")  # default to "text"

        if not question:
            print(f" Skipping item {qid}, missing 'question'.")
            continue

        # Call hybrid agent with proper keys
        result = run_hybrid_agent(
            question=question,
            format_hint=format_hint,
            checkpoint_config={"thread_id": f"thread_{qid}"}
        )

        result["id"] = qid
        outputs.append(result)

    print(f" Saving results to: {out}")
    with open(out, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(" Done. Outputs saved.")


if __name__ == "__main__":
    main()

import time
import os
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class Model:
    def __init__(self):
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        model_name = "sshleifer/distilbart-cnn-12-6"

        print("Loading summarizer model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    def grab_sum(self, eula):
        summary = self.chunk_and_summarize(eula)
        risk = self.assess_risk(eula)
        return summary, risk

    def assess_risk(self, text):
        """
        Returns a risk score from 0–100 based on concerning patterns.
        """
        risk_patterns = {
            r"(collect|track|store).*?(data|information)": 15,
            r"(third[- ]?party|affiliate|partner).*?(share|disclose)": 20,
            r"(may not|must not|prohibited|restricted|forbidden)": 10,
            r"(waive|you agree not to)": 20,
            r"(reverse engineer|circumvent|bypass|vpn)": 10,
            r"(automated|bot|scraper|crawler)": 5,
            r"(location|geo|ip address).*?(track|monitor)": 10,
        }

        score = 0
        lowered = text.lower()

        for pattern, weight in risk_patterns.items():
            if re.search(pattern, lowered, re.IGNORECASE):
                score += weight

        return min(score, 100)
    def _format_summary(self, text):
        sentences = re.findall(r'[^.!?]+[.!?]', text)
        filtered = []
        for s in sentences:
            s = s.strip()
            if len(s.split()) < 6:
                continue
            if s.lower() in [x.lower() for x in filtered]:
                continue
            filtered.append(f"• {s}")
        return "\n".join(filtered)
    def chunk_and_summarize(self, full_text):
        anchors = {
            "Consent & Agreement": r"(agree|consent|click|registration|terms of use)",
            "Service Scope": r"(personal use|noncommercial|individual access)",
            "Usage Restrictions": r"(may not|must not|prohibited|restricted|forbidden|unauthorized)",
            "Data Collection": r"(collect|gather|obtain|track|store|log)",
            "Third-Party Sharing": r"(shared with|provided to|disclosed to|partners|affiliates)",
            "DRM & Circumvention": r"(reverse engineer|circumvent|vpn|tamper|bypass|scraper|bot)"
        }

        chunks = {label: [] for label in anchors}
        lines = full_text.splitlines()

        for line in lines:
            for label, pattern in anchors.items():
                if re.search(pattern, line, re.IGNORECASE):
                    chunks[label].append(line.strip())
                    break

        summarized_sections = []
        for label, lines in chunks.items():
            if not lines:
                continue

            chunk_text = "\n".join(lines)
            inputs = self.tokenizer(
                chunk_text[:3000],
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=20,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

            raw_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            formatted = self._format_summary(raw_summary)
            summarized_sections.append(f"\n🔹 {label}\n{formatted}")

        return "\n".join(summarized_sections)
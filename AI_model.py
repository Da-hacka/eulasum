import time
import os
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Model:
    def __init__(self):
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        def loading_dialogue():
            steps = [
                "Initializing summarizer core...",
                "Loading tokenizer and model weights...",
                "️Configuring device and optimization settings...",
                "Encoding input text for neural digestion...",
                "Summoning beam search and length penalty spells...",
                "Decoding summary from latent space...",
                "Finalizing output and polishing punctuation..."
            ]
            for step in steps:
                print(step)
                time.sleep(0.4)

        model_name = "sshleifer/distilbart-cnn-12-6"
        loading_dialogue()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def grab_sum(self, eula):
        return self.chunk_and_summarize(eula)

    def _format_summary(self, text):
        sentences = re.findall(r'[^.!?]+[.!?]', text)
        filtered = []
        for s in sentences:
            s = s.strip()
            if len(s.split()) < 6 or s.lower() in [x.lower() for x in filtered]:
                continue
            filtered.append(f"• {s}")
        return "\n".join(filtered)

    def _enrich_summary(self, summary, full_text):
        details = self.extract_details(full_text)
        enriched = self._format_summary(summary)

        def expand_clause(clause):
            clause = clause.lower()
            if "password" in clause or "share" in clause:
                return "Accounts must not be shared outside your household. Disney may monitor and enforce this."
            if "commercial" in clause:
                return "Use of the Services for commercial purposes is strictly prohibited."
            if "vpn" in clause or "circumvent" in clause:
                return "Using VPNs or tools to bypass geographic or subscription restrictions is not allowed."
            if "automated" in clause or "bot" in clause:
                return "Automated access via bots or scrapers is forbidden."
            if "reverse engineer" in clause:
                return "Tampering with or reverse engineering the platform or its DRM is prohibited."
            if "geo" in clause or "location" in clause:
                return "Access may be restricted based on geographic location or IP address."
            return clause.strip().capitalize()

        def format_section(title, items):
            if not items:
                return ""
            clean_items = []
            for item in items:
                item = re.sub(r'^[\-\•\s]+', '', item).strip()
                if len(item.split()) < 5:
                    continue
                clean_items.append(expand_clause(item))
            if not clean_items:
                return ""
            return f"\n\n{title}:\n" + "\n".join(f"  - {item}" for item in clean_items)

        enriched += format_section("📦 Data Collected", details["data_collected"])
        enriched += format_section("🚫 Usage Restrictions", details["restrictions"])
        enriched += format_section("⚖️ Rights Waived", details["rights_waived"])
        enriched += format_section("🔗 Third-Party Sharing", details["third_parties"])

        return enriched

    def _group_summary(self, summary_text):
        lines = summary_text.split("\n")
        grouped = {
            "📝 Consent & Agreement": [],
            "📄 Service Scope": [],
            "🚫 Restrictions": [],
            "📦 Data & Sharing": [],
        }
        for line in lines:
            l = line.lower()
            if "agree" in l or "click" in l or "registration" in l:
                grouped["📝 Consent & Agreement"].append(line)
            elif "noncommercial" in l or "personal use" in l or "services are provided" in l:
                grouped["📄 Service Scope"].append(line)
            elif "may not" in l or "prohibited" in l or "restricted" in l or "forbidden" in l:
                grouped["🚫 Restrictions"].append(line)
            elif "data" in l or "third party" in l or "shared" in l or "collected" in l:
                grouped["📦 Data & Sharing"].append(line)
            else:
                grouped["📄 Service Scope"].append(line)

        return "\n".join(
            f"\n{section}\n" + "\n".join(items)
            for section, items in grouped.items() if items
        )

    def extract_details(self, text):
        data_collected = re.findall(
            r"(?:collect|gather|obtain).*?(?:data|information).*?(?:such as|including)?(.*?)[\.\n]",
            text, re.IGNORECASE
        )
        restrictions = re.findall(
            r"(?:may not|must not|prohibited|restricted|forbidden|not permitted).*?(?:use|access|share|stream|distribute|modify|circumvent|reverse engineer|automated|bot).*?[\.\n]",
            text, re.IGNORECASE
        )
        rights_waived = re.findall(
            r"(?:you waive|you agree not to).*?[\.\n]",
            text, re.IGNORECASE
        )
        third_parties = re.findall(
            r"(?:shared with|provided to|disclosed to).*?(?:third[- ]?parties|partners).*?[\.\n]",
            text, re.IGNORECASE
        )

        return {
            "data_collected": [item.strip() for item in data_collected if item.strip()],
            "restrictions": [item.strip() for item in restrictions],
            "rights_waived": [item.strip() for item in rights_waived],
            "third_parties": [item.strip() for item in third_parties]
        }

    def process_eulas(self, eula_list):
        return [self.grab_sum(eula) for eula in eula_list]

    def chunk_and_summarize(self, full_text):
        # Define thematic anchors
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

        # Assign lines to thematic buckets
        for line in lines:
            for label, pattern in anchors.items():
                if re.search(pattern, line, re.IGNORECASE):
                    chunks[label].append(line.strip())
                    break

        # Summarize each chunk
        summarized_sections = []
        for label, lines in chunks.items():
            if not lines:
                continue
            chunk_text = "\n".join(lines)
            inputs = self.tokenizer(chunk_text[:3000], return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=20,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
            raw_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            formatted = self._format_summary(raw_summary)
            summarized_sections.append(f"\n🔹 {label}\n{formatted}")

        return "\n".join(summarized_sections)
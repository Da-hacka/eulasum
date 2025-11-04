from flask import Flask, request, render_template_string
from AI_model import Model
from bs4 import BeautifulSoup
import requests

class Gui:
    def __init__(self):
        self.model = Model()
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/", methods=["GET", "POST"])
        def home():
            summary = ""
            if request.method == "POST":
                raw_input = request.form.get("eula_input", "").strip()
                try:
                    if self.is_url(raw_input):
                        eula_text = self.fetch_clean_text(raw_input)
                    else:
                        eula_text = raw_input
                    summary = self.model.grab_sum(eula=eula_text)
                except Exception as e:
                    summary = f"Error during summarization:\n{e}"
            return self.render_page(summary)

    def run(self):
        self.app.run(debug=True)

    def render_page(self, summary_text):
        html = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        .smooth-button {
          background-color: #393E46; 
          color: white; 
          border: none;
          padding: 10px 20px;
          font-size: 16px; 
          border-radius: 5px;
          cursor: pointer;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
          transition: all 0.3s ease; 
        }

        .smooth-button:hover {
          background-color: #45a049; 
          box-shadow: 0 6px 8px rgba(0, 0, 0, 0.3); 
          transform: scale(1.05);
        }

        .smooth-button:active {
          transform: scale(0.95); 
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); 
        }

        .spinner {
          border: 6px solid #f3f3f3;
          border-top: 6px solid #00ADB5;
          border-radius: 50%;
          width: 40px;
          height: 40px;
          animation: spin 1s linear infinite;
          margin: 10px auto;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        </style>
        <script>
        document.addEventListener("DOMContentLoaded", function() {
          const form = document.querySelector("form");
          form.addEventListener("submit", () => {
            document.getElementById("loading").style.display = "block";
          });
        });
        </script>
        <link href="https://fonts.googleapis.com/css2?family=Comfortaa:wght@400;700&display=swap" rel="stylesheet">
        <title>EULA Summarizer</title>
        </head>
        <body style="background-color:#222831; color:#DFD0B8; font-family: 'Comfortaa', sans-serif;">
            <h1>EULASum: Summarize with Ease</h1>
            <form method="post">
                <label>Enter EULA text or URL:</label><br>
                <textarea name="eula_input" rows="10" cols="80" style="background-color:#393E46; color:#DFD0B8;"></textarea><br><br>
                <button class="smooth-button" type="submit">Generate Summary</button>
            </form>
            <div id="loading" style="display:none;">
                <p>Summarizing... please wait</p>
                <div class="spinner"></div>
            </div>
            <h2>Summary Output:</h2>
            <textarea readonly rows="15" cols="80" style="background-color:#393E46; color:#DFD0B8;">{{ summary_text }}</textarea>
        </body>
        </html>
        """
        return render_template_string(html, summary_text=summary_text)

    def is_url(self, text):
        return text.strip().lower().startswith("http")

    def fetch_clean_text(self, url):
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator="\n")
export default function Footer() {
  return (
    <footer className="footer">
      <p>
        SurveyIQ · Built by Rayan Malik, Kaustubh Bukkapatnam &amp; Pranav Gadde ·
        IMSA · SAIL AI Innovation Fellowship
      </p>
      <p className="footer-note">
        Uploaded responses are analyzed in memory and discarded. If you name a survey,
        only its date, response count, quality score and per-question averages are kept
        for trend tracking — never individual answers or respondent details.
      </p>
    </footer>
  );
}

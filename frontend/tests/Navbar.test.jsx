import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import Navbar from "../src/components/Navbar.jsx";
import SignalsSection from "../src/components/SignalsSection.jsx";
import Hero from "../src/components/Hero.jsx";
import Footer from "../src/components/Footer.jsx";

describe("Navbar", () => {
  it("shows all three steps", () => {
    render(<Navbar step="upload" />);
    expect(screen.getByText("Upload")).toBeInTheDocument();
    expect(screen.getByText("Map columns")).toBeInTheDocument();
    expect(screen.getByText("Results")).toBeInTheDocument();
  });

  it("marks the current step for assistive tech", () => {
    render(<Navbar step="map" />);
    const current = document.querySelector('[aria-current="step"]');
    expect(current).toHaveTextContent("Map columns");
  });

  it("marks earlier steps as done and later ones as not", () => {
    render(<Navbar step="results" />);
    const done = [...document.querySelectorAll(".step.done")].map((e) => e.textContent);
    expect(done.join(" ")).toContain("Upload");
    expect(done.join(" ")).toContain("Map columns");
    expect(document.querySelectorAll(".step.todo")).toHaveLength(0);
  });
});

describe("Hero", () => {
  it("states what the product does", () => {
    render(<Hero />);
    expect(screen.getByRole("heading", { level: 1 })).toBeInTheDocument();
    expect(screen.getByText(/plain-English reason/i)).toBeInTheDocument();
  });
});

describe("SignalsSection", () => {
  it("lists all six detection signals", () => {
    render(<SignalsSection />);
    for (const name of [
      "Straightlining", "Speeding", "Random answering",
      "Contradictions", "Attention checks", "Extreme responding",
    ]) {
      expect(screen.getByText(name)).toBeInTheDocument();
    }
  });

  it("states the explainability promise accurately", () => {
    render(<SignalsSection />);
    // must say decision tree -- the model we actually ship
    expect(screen.getByText(/decision tree/i)).toBeInTheDocument();
  });
});

describe("Footer", () => {
  it("credits the team and states the privacy posture accurately", () => {
    render(<Footer />);
    expect(screen.getByText(/Rayan Malik/)).toBeInTheDocument();
    // Responses are discarded, but named runs keep aggregate totals for trends --
    // the footer must describe what actually happens, not an older simpler claim.
    expect(screen.getByText(/analyzed in memory and discarded/i)).toBeInTheDocument();
    expect(screen.getByText(/never answers or respondent details/i)).toBeInTheDocument();
  });
});

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import UploadStep from "../src/components/UploadStep.jsx";

function csvFile(name = "survey.csv") {
  return new File(["a,b\n1,2\n"], name, { type: "text/csv" });
}

describe("UploadStep", () => {
  it("disables the action until a file is chosen", () => {
    render(<UploadStep onFileReady={() => {}} busy={false} error="" />);
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
  });

  it("enables the action and reports the file once chosen", () => {
    render(<UploadStep onFileReady={() => {}} busy={false} error="" />);
    fireEvent.change(screen.getByLabelText(/survey file/i), {
      target: { files: [csvFile()] },
    });
    expect(screen.getByText(/survey\.csv/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();
  });

  it("passes the chosen file to onFileReady", () => {
    const onFileReady = vi.fn();
    render(<UploadStep onFileReady={onFileReady} busy={false} error="" />);
    const file = csvFile("mine.csv");
    fireEvent.change(screen.getByLabelText(/survey file/i), { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));
    expect(onFileReady).toHaveBeenCalledWith(file);
  });

  it("shows an error message when given one", () => {
    render(<UploadStep onFileReady={() => {}} busy={false} error="could not parse CSV" />);
    expect(screen.getByText(/could not parse csv/i)).toBeInTheDocument();
  });
});

describe("UploadStep sample survey", () => {
  it("offers a sample survey without requiring a file", () => {
    render(<UploadStep onFileReady={() => {}} onTrySample={() => {}} busy={false} error="" />);
    const sample = screen.getByRole("button", { name: /sample survey/i });
    expect(sample).toBeEnabled();
    // the primary action still needs a file
    expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled();
  });

  it("calls onTrySample when clicked", () => {
    const onTrySample = vi.fn();
    render(<UploadStep onFileReady={() => {}} onTrySample={onTrySample} busy={false} error="" />);
    fireEvent.click(screen.getByRole("button", { name: /sample survey/i }));
    expect(onTrySample).toHaveBeenCalled();
  });

  it("accepts spreadsheet files too", () => {
    render(<UploadStep onFileReady={() => {}} onTrySample={() => {}} busy={false} error="" />);
    expect(screen.getByLabelText(/survey file/i).getAttribute("accept")).toContain(".xlsx");
  });
});

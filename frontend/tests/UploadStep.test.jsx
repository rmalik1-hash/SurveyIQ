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
    fireEvent.change(screen.getByLabelText(/survey csv file/i), {
      target: { files: [csvFile()] },
    });
    expect(screen.getByText(/survey\.csv/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled();
  });

  it("passes the chosen file to onFileReady", () => {
    const onFileReady = vi.fn();
    render(<UploadStep onFileReady={onFileReady} busy={false} error="" />);
    const file = csvFile("mine.csv");
    fireEvent.change(screen.getByLabelText(/survey csv file/i), { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));
    expect(onFileReady).toHaveBeenCalledWith(file);
  });

  it("shows an error message when given one", () => {
    render(<UploadStep onFileReady={() => {}} busy={false} error="could not parse CSV" />);
    expect(screen.getByText(/could not parse csv/i)).toBeInTheDocument();
  });
});

import { Component } from "react";

/**
 * Keeps a failing chart from taking down the whole dashboard. The scores and
 * explanations matter far more than the pictures, so a chart that cannot draw
 * degrades to a short message instead of unmounting everything.
 */
export default class ChartBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { failed: false };
  }

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error) {
    console.error("Chart failed to render:", error);
  }

  render() {
    if (this.state.failed) {
      return <p className="hint">This chart could not be displayed.</p>;
    }
    return this.props.children;
  }
}

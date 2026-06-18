import io
import streamlit as st

def add_download_button(fig, filename_prefix="figure", plot_type="matplotlib", fmt="svg", dpi=300, scale=3):
    """Streamlit download button for:"""
    plot_type = str(plot_type).lower().strip()
    fmt = str(fmt).lower().strip()

    if fmt not in {"png", "svg"}:
        raise ValueError("fmt must be 'png' or 'svg'")

    if plot_type == "matplotlib":
        buf = io.BytesIO()

        if fmt == "png":
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            mime = "image/png"
        else:
            fig.savefig(buf, format="svg", bbox_inches="tight")
            mime = "image/svg+xml"

        buf.seek(0)
        st.download_button(
            label=f"📥 Download {filename_prefix}.{fmt}",
            data=buf.getvalue(),
            file_name=f"{filename_prefix}.{fmt}",
            mime=mime,
        )
        return

    if plot_type == "plotly":
        try:
            img_bytes = fig.to_image(format=fmt, scale=scale)
        except Exception as e:
            st.warning(
                f"Static export to {fmt.upper()} requires kaleido. "
                f"Install with: pip install -U kaleido\n\n"
                f"Export error: {e}"
            )
            return

        mime = "image/png" if fmt == "png" else "image/svg+xml"
        st.download_button(
            label=f"📥 Download {filename_prefix}.{fmt}",
            data=img_bytes,
            file_name=f"{filename_prefix}.{fmt}",
            mime=mime,
        )
        return

    raise ValueError("plot_type must be 'matplotlib' or 'plotly'")

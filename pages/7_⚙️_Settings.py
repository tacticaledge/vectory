"""Settings Page - Theme Configuration, API Keys, and Preferences"""

import streamlit as st
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.models import init_session_state
from components.model_catalog import get_provider_models
from components.themes import THEMES, get_theme, get_theme_css
from components.ui import sidebar_logo


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to RGB values for rgba() usage."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"{r}, {g}, {b}"


st.set_page_config(page_title="Settings | Vectory", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")
init_session_state(st)

# Get current theme and inject CSS
current_theme = get_theme(st.session_state.get("theme", "dark"))
st.markdown(get_theme_css(current_theme), unsafe_allow_html=True)

# Header
st.markdown(f"""
<h1 style="font-family: 'DM Sans', sans-serif; background: linear-gradient(135deg, {current_theme.gradient_start}, {current_theme.gradient_end}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem;">
    ⚙️ Settings
</h1>
<p style="font-family: 'DM Sans', sans-serif; color: {current_theme.text_muted}; font-size: 1rem;">
    <span style="color: {current_theme.terminal};">$</span> configure your Vectory experience
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# API Keys Section
# ============================================================================
st.markdown(f"""
<h3 style="color: {current_theme.text_primary}; font-family: 'DM Sans', sans-serif;">
    🔑 API Keys
</h3>
    <p style="color: {current_theme.text_muted}; font-size: 0.9rem;">
    Configure provider keys for LLM-as-Judge and taxonomy workflows. Keys are stored in this browser session only.
</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div style="background: {current_theme.bg_card}; border: 1px solid {current_theme.border}; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
        <h4 style="margin: 0 0 12px 0; color: {current_theme.text_primary}; font-family: 'DM Sans', sans-serif;">
            OpenAI
        </h4>
        <p style="color: {current_theme.text_muted}; font-size: 0.85rem; margin-bottom: 12px;">
            Supports current OpenAI evaluation models including GPT-5.1, GPT-5 mini, GPT-5 nano, and GPT-4.1.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Get existing key from session or environment
    openai_key = st.session_state.get("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))
    new_openai_key = st.text_input(
        "OpenAI API Key",
        value=openai_key,
        type="password",
        key="openai_key_input",
        placeholder="Enter OpenAI API key",
        label_visibility="collapsed"
    )
    if new_openai_key != openai_key:
        st.session_state.openai_api_key = new_openai_key
        if new_openai_key:
            st.success("OpenAI API key saved to session")

with col2:
    st.markdown(f"""
    <div style="background: {current_theme.bg_card}; border: 1px solid {current_theme.border}; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
        <h4 style="margin: 0 0 12px 0; color: {current_theme.text_primary}; font-family: 'DM Sans', sans-serif;">
            Anthropic
        </h4>
        <p style="color: {current_theme.text_muted}; font-size: 0.85rem; margin-bottom: 12px;">
            Supports configured Anthropic evaluation models for complex, balanced, and fast judging.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Get existing key from session or environment
    anthropic_key = st.session_state.get("anthropic_api_key", os.environ.get("ANTHROPIC_API_KEY", ""))
    new_anthropic_key = st.text_input(
        "Anthropic API Key",
        value=anthropic_key,
        type="password",
        key="anthropic_key_input",
        placeholder="Enter Anthropic API key",
        label_visibility="collapsed"
    )
    if new_anthropic_key != anthropic_key:
        st.session_state.anthropic_api_key = new_anthropic_key
        if new_anthropic_key:
            st.success("Anthropic API key saved to session")

st.markdown(f"""
<p style="color: {current_theme.text_muted}; font-size: 0.8rem; margin-top: 8px;">
    💡 <strong>Tip:</strong> For persistent local development storage, create <code style="background: {current_theme.code_bg}; padding: 2px 6px; border-radius: 4px;">.streamlit/secrets.toml</code> with your keys. Do not commit that file.
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"<hr style='border-color: {current_theme.border};'>", unsafe_allow_html=True)

# ============================================================================
# Local Install Section
# ============================================================================
st.markdown(f"""
<h3 style="color: {current_theme.text_primary}; font-family: 'DM Sans', sans-serif;">
    💻 Local CLI Install
</h3>
<p style="color: {current_theme.text_muted}; font-size: 0.9rem;">
    Install Vectory from PyPI with pip or pipx, then launch the local evaluation app.
</p>
""", unsafe_allow_html=True)

install_cols = st.columns(3)

with install_cols[0]:
    st.markdown(f"""
    <div style="background: {current_theme.bg_card}; border: 1px solid {current_theme.border}; border-radius: 12px; padding: 18px; min-height: 128px;">
        <h4 style="margin: 0 0 8px 0; color: {current_theme.text_primary}; font-family: 'DM Sans', sans-serif;">Install with pip</h4>
        <p style="margin: 0; color: {current_theme.text_muted}; font-size: 0.84rem;">Install into an activated virtual environment or project environment.</p>
    </div>
    """, unsafe_allow_html=True)
    st.code("pip install vectoryai", language="bash")

with install_cols[1]:
    st.markdown(f"""
    <div style="background: {current_theme.bg_card}; border: 1px solid {current_theme.border}; border-radius: 12px; padding: 18px; min-height: 128px;">
        <h4 style="margin: 0 0 8px 0; color: {current_theme.text_primary}; font-family: 'DM Sans', sans-serif;">Install with pipx</h4>
        <p style="margin: 0; color: {current_theme.text_muted}; font-size: 0.84rem;">Use pipx for an isolated CLI without mixing dependencies into another project.</p>
    </div>
    """, unsafe_allow_html=True)
    st.code("pipx install vectoryai", language="bash")

with install_cols[2]:
    st.markdown(f"""
    <div style="background: {current_theme.bg_card}; border: 1px solid {current_theme.border}; border-radius: 12px; padding: 18px; min-height: 128px;">
        <h4 style="margin: 0 0 8px 0; color: {current_theme.text_primary}; font-family: 'DM Sans', sans-serif;">Run</h4>
        <p style="margin: 0; color: {current_theme.text_muted}; font-size: 0.84rem;">Launch the local app with the installed CLI command.</p>
    </div>
    """, unsafe_allow_html=True)
    st.code("vectory app", language="bash")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"<hr style='border-color: {current_theme.border};'>", unsafe_allow_html=True)

# ============================================================================
# Model Catalog Section
# ============================================================================
st.markdown(f"""
<h3 style="color: {current_theme.text_primary}; font-family: 'DM Sans', sans-serif;">
    🧠 Model Catalog
</h3>
<p style="color: {current_theme.text_muted}; font-size: 0.9rem;">
    Current provider defaults used by evaluation pages. Refresh date: July 26, 2026.
</p>
""", unsafe_allow_html=True)

catalog_col1, catalog_col2 = st.columns(2)
for provider, column in [("openai", catalog_col1), ("anthropic", catalog_col2)]:
    with column:
        provider_label = "OpenAI" if provider == "openai" else "Anthropic"
        model_rows = []
        for model in get_provider_models(provider):
            model_rows.append(f"""
            <div style="padding: 10px 0; border-top: 1px solid {current_theme.border};">
                <div style="font-weight: 700; color: {current_theme.text_primary};">{model.label}</div>
                <div style="font-size: 0.78rem; color: {current_theme.text_muted}; font-family: 'DM Sans', sans-serif;">{model.model_id}</div>
                <div style="font-size: 0.82rem; color: {current_theme.text_secondary}; margin-top: 4px;">{model.note}</div>
            </div>
            """)
        if not model_rows:
            model_rows.append(f"""
            <div style="padding: 10px 0; border-top: 1px solid {current_theme.border};">
                <div style="font-size: 0.82rem; color: {current_theme.text_secondary};">No public default models configured.</div>
            </div>
            """)
        st.markdown(f"""
        <div style="background: {current_theme.bg_card}; border: 1px solid {current_theme.border}; border-radius: 8px; padding: 18px; margin-bottom: 12px;">
            <h4 style="margin: 0 0 12px 0; color: {current_theme.text_primary}; font-family: 'DM Sans', sans-serif;">{provider_label}</h4>
            {''.join(model_rows)}
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"<hr style='border-color: {current_theme.border};'>", unsafe_allow_html=True)

# ============================================================================
# Theme Selection Section
# ============================================================================
st.markdown(f"""
<h3 style="color: {current_theme.text_primary}; font-family: 'DM Sans', sans-serif;">
    🎨 Theme
</h3>
<p style="color: {current_theme.text_muted}; font-size: 0.9rem;">
    Choose a theme that works best for you. Changes apply immediately.
</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    # Theme cards
    for theme_name, theme in THEMES.items():
        is_selected = st.session_state.get("theme", "dark") == theme_name

        # Create theme preview card
        border_color = theme.primary if is_selected else theme.border
        check_mark = f'<span style="color: {theme.success}; margin-left: 8px;">✓</span>' if is_selected else ''

        st.markdown(f"""
        <div style="
            background: {theme.bg_card};
            border: 2px solid {border_color};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 12px;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="margin: 0; color: {theme.text_primary}; font-family: 'DM Sans', sans-serif; font-weight: 600; font-size: 1rem;">
                        {theme.display_name}{check_mark}
                    </div>
                    <div style="margin: 8px 0 0 0; color: {theme.text_muted}; font-size: 0.85rem;">
                        {theme.description}
                    </div>
                </div>
                <div style="display: flex; gap: 6px;">
                    <div style="width: 20px; height: 20px; border-radius: 50%; background: {theme.primary}; border: 1px solid {theme.border};"></div>
                    <div style="width: 20px; height: 20px; border-radius: 50%; background: {theme.secondary}; border: 1px solid {theme.border};"></div>
                    <div style="width: 20px; height: 20px; border-radius: 50%; background: {theme.accent}; border: 1px solid {theme.border};"></div>
                    <div style="width: 20px; height: 20px; border-radius: 50%; background: {theme.success}; border: 1px solid {theme.border};"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            f"{'✓ Active' if is_selected else 'Apply'} ",
            key=f"theme_{theme_name}",
            disabled=is_selected,
            use_container_width=True
        ):
            st.session_state.theme = theme_name
            st.rerun()

with col2:
    st.markdown(f"""
    <div style="background: {current_theme.bg_card}; border: 1px solid {current_theme.border}; border-radius: 12px; padding: 20px;">
        <h4 style="margin: 0 0 16px 0; color: {current_theme.text_primary}; font-family: 'DM Sans', sans-serif;">
            Preview
        </h4>
        <div style="
            background: linear-gradient(135deg, {current_theme.gradient_start}, {current_theme.gradient_mid}, {current_theme.gradient_end});
            padding: 16px;
            border-radius: 8px;
            color: white;
            text-align: center;
            font-family: 'DM Sans', sans-serif;
            margin-bottom: 12px;
            font-size: 0.9rem;
        ">
            Gradient Preview
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px;">
            <span style="background: rgba({_hex_to_rgb(current_theme.success)}, 0.2); color: {current_theme.success}; padding: 3px 8px; border-radius: 4px; font-size: 0.7rem; font-family: 'DM Sans', sans-serif; border: 1px solid {current_theme.success};">SUCCESS</span>
            <span style="background: rgba({_hex_to_rgb(current_theme.warning)}, 0.2); color: {current_theme.warning}; padding: 3px 8px; border-radius: 4px; font-size: 0.7rem; font-family: 'DM Sans', sans-serif; border: 1px solid {current_theme.warning};">WARNING</span>
            <span style="background: rgba({_hex_to_rgb(current_theme.error)}, 0.2); color: {current_theme.error}; padding: 3px 8px; border-radius: 4px; font-size: 0.7rem; font-family: 'DM Sans', sans-serif; border: 1px solid {current_theme.error};">ERROR</span>
            <span style="background: rgba({_hex_to_rgb(current_theme.info)}, 0.2); color: {current_theme.info}; padding: 3px 8px; border-radius: 4px; font-size: 0.7rem; font-family: 'DM Sans', sans-serif; border: 1px solid {current_theme.info};">INFO</span>
        </div>
        <div style="
            background: {current_theme.code_bg};
            border: 1px solid {current_theme.border};
            border-radius: 8px;
            padding: 10px;
            font-family: 'DM Sans', sans-serif;
            font-size: 0.75rem;
        ">
            <span style="color: {current_theme.terminal};">$</span>
            <span style="color: {current_theme.text_secondary};"> vectory --theme {st.session_state.get('theme', 'dark')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"<hr style='border-color: {current_theme.border};'>", unsafe_allow_html=True)

# ============================================================================
# Accessibility Information
# ============================================================================
st.markdown(f"""
<h3 style="color: {current_theme.text_primary}; font-family: 'DM Sans', sans-serif;">
    ♿ Accessibility
</h3>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background: {current_theme.bg_card}; border: 1px solid {current_theme.border}; border-radius: 12px; padding: 20px;">
    <h4 style="margin: 0 0 12px 0; color: {current_theme.text_primary}; font-family: 'DM Sans', sans-serif;">Theme Accessibility Features</h4>
    <ul style="color: {current_theme.text_secondary}; line-height: 1.8; margin: 0; padding-left: 20px;">
        <li><strong style="color: {current_theme.text_primary};">Dark Developer</strong> - Optimized for long coding sessions with reduced eye strain</li>
        <li><strong style="color: {current_theme.text_primary};">Light</strong> - High readability in bright environments</li>
        <li><strong style="color: {current_theme.text_primary};">High Contrast (Colorblind Safe)</strong> - Uses a blue-orange palette distinguishable by most forms of color blindness</li>
        <li><strong style="color: {current_theme.text_primary};">High Contrast Dark</strong> - Maximum contrast for users with low vision</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"<hr style='border-color: {current_theme.border};'>", unsafe_allow_html=True)

# ============================================================================
# About Section
# ============================================================================
st.markdown(f"""
<h3 style="color: {current_theme.text_primary}; font-family: 'DM Sans', sans-serif;">
    ℹ️ About
</h3>
""", unsafe_allow_html=True)

from components import __version__, __product_name__, __tagline__, __author__, __website__

st.markdown(f"""
<div style="background: {current_theme.bg_card}; border: 1px solid {current_theme.border}; border-radius: 12px; padding: 20px;">
    <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 16px;">
        <span style="font-size: 2.5rem; font-weight: 800; font-family: 'DM Sans', sans-serif; background: linear-gradient(135deg, {current_theme.gradient_start}, {current_theme.gradient_end}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">V</span>
        <div>
            <h3 style="margin: 0; color: {current_theme.text_primary}; font-family: 'DM Sans', sans-serif;">{__product_name__}</h3>
            <p style="margin: 4px 0 0 0; color: {current_theme.text_muted}; font-size: 0.85rem;">{__tagline__}</p>
        </div>
    </div>
    <p style="color: {current_theme.text_secondary}; margin-bottom: 8px; font-size: 0.9rem;">
        <strong>Version:</strong> <span style="color: {current_theme.accent}; font-family: 'DM Sans', sans-serif;">v{__version__}</span>
    </p>
    <p style="color: {current_theme.text_secondary}; margin-bottom: 8px; font-size: 0.9rem;">
        <strong>Developer:</strong> <a href="{__website__}" target="_blank" style="color: {current_theme.primary};">{__author__}</a>
    </p>
    <p style="color: {current_theme.text_muted}; font-size: 0.85rem; margin-top: 12px;">
        Vectory is a professional-grade toolkit for evaluating Large Language Model outputs.
    </p>
</div>
""", unsafe_allow_html=True)

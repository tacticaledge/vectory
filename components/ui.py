"""
Vectory - Animated UI components for the LLM Evaluation platform.
Uses configurable themes with Lottie animations and custom CSS.
"""

import streamlit as st
from typing import Optional

from components.themes import get_theme, get_theme_css, THEMES, DEFAULT_THEME

# Lottie animation URLs (free animations from LottieFiles)
LOTTIE_URLS = {
    "robot": "https://assets5.lottiefiles.com/packages/lf20_M9p23l.json",
    "data": "https://assets3.lottiefiles.com/packages/lf20_qp1q7mct.json",
    "chart": "https://assets9.lottiefiles.com/packages/lf20_kd5rzej5.json",
    "loading": "https://assets2.lottiefiles.com/packages/lf20_p8bfn5to.json",
    "success": "https://assets4.lottiefiles.com/packages/lf20_jbrw3hcz.json",
    "search": "https://assets10.lottiefiles.com/packages/lf20_hy4txm7l.json",
    "ai": "https://assets2.lottiefiles.com/packages/lf20_oyi9a28g.json",
    "rocket": "https://assets9.lottiefiles.com/packages/lf20_l13zwziy.json",
    "trophy": "https://assets1.lottiefiles.com/packages/lf20_touohxv0.json",
    "check": "https://assets8.lottiefiles.com/packages/lf20_uu0x8lqv.json",
}


def get_current_theme():
    """Get the current theme from session state."""
    theme_name = st.session_state.get("theme", DEFAULT_THEME)
    return get_theme(theme_name)


def load_lottie_url(url: str) -> Optional[dict]:
    """Load Lottie animation from URL."""
    try:
        import requests
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def inject_custom_css():
    """Inject custom CSS based on current theme."""
    theme = get_current_theme()
    st.markdown(get_theme_css(theme), unsafe_allow_html=True)


def sidebar_logo():
    """Render the Vectory logo in the sidebar at the top left."""
    theme = get_current_theme()
    with st.sidebar:
        st.markdown(f'''<div style="display:flex;align-items:center;gap:10px;padding:8px 0 16px 0;border-bottom:1px solid {theme.border};margin-bottom:12px;"><div style="width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,{theme.gradient_start},{theme.gradient_end});display:flex;align-items:center;justify-content:center;"><span style="font-size:16px;font-weight:800;color:white;">V</span></div><span style="font-size:1.1rem;font-weight:600;color:{theme.text_primary};font-family:'DM Sans',sans-serif;">Vectory</span></div>''', unsafe_allow_html=True)


def animated_metric(label: str, value: str, icon: str = "", delay: int = 0):
    """Display an animated metric card."""
    st.markdown(f"""
    <div class="metric-card delay-{delay}">
        <div style="font-size: 1.5rem; margin-bottom: 8px;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def feature_card(icon: str, title: str, description: str, delay: int = 0):
    """Display an animated feature card."""
    st.markdown(f"""
    <div class="feature-card delay-{delay}">
        <div class="feature-icon">{icon}</div>
        <div class="feature-title">{title}</div>
        <div class="feature-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def hero_section(title: str, subtitle: str):
    """Display an animated hero section."""
    st.markdown(f"""
    <div style="text-align: center; padding: 40px 0;">
        <h1 class="hero-title">{title}</h1>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def gradient_banner(content: str):
    """Display an animated gradient banner using theme colors."""
    theme = get_current_theme()
    st.markdown(f"""
    <div class="themed-banner">
        {content}
    </div>
    """, unsafe_allow_html=True)


def animated_progress_bar():
    """Display an animated progress bar."""
    st.markdown('<div class="animated-progress"></div>', unsafe_allow_html=True)


def score_bar(score: float, max_score: float = 1.0, label: str = ""):
    """Display an animated score bar."""
    theme = get_current_theme()
    percentage = min(100, (score / max_score) * 100)
    st.markdown(f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-family: var(--theme-font-mono);">
            <span style="font-weight: 500; color: var(--theme-text-primary);">{label}</span>
            <span style="color: var(--theme-accent); font-weight: 600;">{score:.3f}</span>
        </div>
        <div class="score-bar-container">
            <div class="score-bar" style="width: {percentage}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def rank_badge(rank: int):
    """Display a rank badge."""
    theme = get_current_theme()
    rank_class = f"rank-{rank}" if rank <= 3 else ""
    style = f"background: {theme.border}; color: {theme.text_primary};" if rank > 3 else ""
    st.markdown(f"""
    <div class="rank-badge {rank_class}" style="{style}">
        {rank}
    </div>
    """, unsafe_allow_html=True)


def animated_list_item(content: str, delay: int = 0):
    """Display an animated list item."""
    st.markdown(f"""
    <div class="animated-list-item delay-{delay}">
        {content}
    </div>
    """, unsafe_allow_html=True)


def loading_shimmer(height: int = 100):
    """Display a shimmer loading placeholder."""
    st.markdown(f"""
    <div class="shimmer" style="height: {height}px; width: 100%;"></div>
    """, unsafe_allow_html=True)


def display_lottie(animation_name: str, height: int = 200, key: str = None):
    """Display a Lottie animation."""
    try:
        from streamlit_lottie import st_lottie

        url = LOTTIE_URLS.get(animation_name)
        if url:
            animation = load_lottie_url(url)
            if animation:
                st_lottie(animation, height=height, key=key or animation_name)
                return True
    except ImportError:
        pass
    return False


def glass_card(content: str):
    """Display a glass-morphism card."""
    st.markdown(f"""
    <div class="glass-card">
        {content}
    </div>
    """, unsafe_allow_html=True)


def vectory_logo(size: str = "large"):
    """Display the Vectory logo."""
    if size == "large":
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div class="vectory-logo" style="font-size: 3rem;">Vectory</div>
            <div class="vectory-tagline">Precision LLM Evaluation</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center;">
            <div class="vectory-logo">Vectory</div>
            <div class="vectory-tagline" style="font-size: 0.7rem;">Precision LLM Evaluation</div>
        </div>
        """, unsafe_allow_html=True)


def animated_logo(size: str = "large", show_text: bool = True):
    """Display an animated Vectory logo representing LLM evaluation."""
    theme = get_current_theme()

    if size == "large":
        logo_size = 120
        text_size = "2.5rem"
        tagline_size = "0.9rem"
    elif size == "medium":
        logo_size = 80
        text_size = "1.8rem"
        tagline_size = "0.75rem"
    else:
        logo_size = 50
        text_size = "1.2rem"
        tagline_size = "0.6rem"

    # Build HTML as a single clean string
    logo_style = f"width:{logo_size}px;height:{logo_size}px;border-radius:50%;background:linear-gradient(135deg,{theme.bg_card} 0%,{theme.bg_secondary} 100%);border:3px solid transparent;background-image:linear-gradient({theme.bg_card},{theme.bg_card}),linear-gradient(135deg,{theme.gradient_start},{theme.gradient_end});background-origin:border-box;background-clip:padding-box,border-box;display:flex;align-items:center;justify-content:center;margin:0 auto;box-shadow:0 0 30px rgba(99,102,241,0.3);"

    v_style = f"font-size:{int(logo_size * 0.45)}px;font-weight:800;background:linear-gradient(135deg,{theme.gradient_start},{theme.gradient_end});-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;"

    html = f'<div style="text-align:center;padding:20px 0;"><div style="{logo_style}"><div style="{v_style}">V</div></div>'

    if show_text:
        html += f'<div class="vectory-logo" style="font-size:{text_size};margin-top:12px;">Vectory</div>'
        html += f'<div class="vectory-tagline" style="font-size:{tagline_size};">Precision LLM Evaluation</div>'

    html += '</div>'

    st.markdown(html, unsafe_allow_html=True)


def sidebar_logo(product_name: str = "Vectory", tagline: str = "Precision LLM Evaluation", version: str = "1.0.0"):
    """Display a compact logo with branding for the sidebar."""
    theme = get_current_theme()

    st.markdown(f'''
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 12px;">
            <div style="
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: linear-gradient(135deg, {theme.gradient_start}, {theme.gradient_end});
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
            ">
                <span style="
                    font-size: 20px;
                    font-weight: 800;
                    color: white;
                    font-family: var(--theme-font-mono);
                ">V</span>
            </div>
            <div>
                <div class="vectory-logo" style="font-size: 1.6rem; text-align: left;">
                    {product_name}
                </div>
                <p class="vectory-tagline" style="text-align: left; margin: 0;">{tagline}</p>
            </div>
        </div>
        <div style="margin-top: 12px;">
            <span style="
                background: linear-gradient(135deg, {theme.primary}, {theme.secondary});
                color: white;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 0.7rem;
                font-family: var(--theme-font-mono);
                font-weight: 500;
            ">v{version}</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)


def app_footer(product_name: str, tagline: str, author: str, website: str, version: str):
    """Display a professional app footer."""
    theme = get_current_theme()
    current_year = __import__('datetime').datetime.now().year

    st.markdown(f'''
    <footer style="
        margin-top: 60px;
        padding: 40px 20px;
        background: linear-gradient(180deg, transparent 0%, {theme.bg_secondary} 30%);
        border-top: 1px solid {theme.border};
    ">
        <div style="max-width: 1200px; margin: 0 auto; text-align: center;">
            <!-- Logo and tagline -->
            <div style="margin-bottom: 24px;">
                <span style="
                    font-family: var(--theme-font-mono);
                    font-size: 1.5rem;
                    font-weight: 700;
                    background: linear-gradient(135deg, {theme.gradient_start}, {theme.gradient_end});
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                ">{product_name}</span>
                <span style="color: {theme.text_muted}; margin: 0 12px;">|</span>
                <span style="color: {theme.text_secondary}; font-family: var(--theme-font-mono);">{tagline}</span>
            </div>

            <!-- Links -->
            <div style="margin-bottom: 20px; display: flex; justify-content: center; gap: 24px; flex-wrap: wrap;">
                <a href="#" style="color: {theme.text_muted}; text-decoration: none; font-size: 0.85rem; transition: color 0.3s;">
                    Documentation
                </a>
                <a href="#" style="color: {theme.text_muted}; text-decoration: none; font-size: 0.85rem;">
                    GitHub
                </a>
                <a href="{website}" target="_blank" style="color: {theme.text_muted}; text-decoration: none; font-size: 0.85rem;">
                    {author}
                </a>
            </div>

            <!-- Copyright -->
            <div style="
                color: {theme.text_muted};
                font-size: 0.75rem;
                font-family: var(--theme-font-mono);
            ">
                <span style="color: {theme.accent};">v{version}</span>
                <span style="margin: 0 8px;">·</span>
                © {current_year} {author}. All rights reserved.
            </div>
        </div>
    </footer>
    ''', unsafe_allow_html=True)


def page_header(title: str, icon: str = ""):
    """Display a consistent page header with Vectory styling."""
    st.markdown(f"""
    <div class="page-header">
        <h1 style="margin: 0; font-size: 2rem;">
            {icon + " " if icon else ""}{title}
        </h1>
    </div>
    """, unsafe_allow_html=True)


def terminal_output(text: str):
    """Display text in terminal style."""
    st.markdown(f"""
    <div class="terminal-box">
        <span class="terminal-prompt">{text}</span>
    </div>
    """, unsafe_allow_html=True)


def code_block(code: str, language: str = "python"):
    """Display a styled code block."""
    st.markdown(f"""
    <div class="code-block">
        <pre><code>{code}</code></pre>
    </div>
    """, unsafe_allow_html=True)


def status_badge(text: str, status: str = "info"):
    """Display a status badge. Status can be: success, warning, error, info."""
    st.markdown(f"""
    <span class="status-badge status-{status}">{text}</span>
    """, unsafe_allow_html=True)


def dev_stat(label: str, value: str):
    """Display a developer-style statistic."""
    st.markdown(f"""
    <div class="dev-stat">
        <div class="dev-stat-label">{label}</div>
        <div class="dev-stat-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def themed_header(title: str, subtitle: str):
    """Display a themed page header."""
    theme = get_current_theme()
    st.markdown(f"""
    <h1 class="themed-header">{title}</h1>
    <p class="themed-subtitle">{subtitle}</p>
    """, unsafe_allow_html=True)


def themed_card(content: str):
    """Display a themed card."""
    st.markdown(f"""
    <div class="themed-card">
        {content}
    </div>
    """, unsafe_allow_html=True)


def _escape_html(text: str) -> str:
    """Escape HTML special characters to prevent injection."""
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to RGB values for rgba() usage."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"{r}, {g}, {b}"


# Valid style options for section_header
_VALID_STYLES = frozenset({"primary", "success", "warning", "info", "error", "accent"})


def section_header(title: str, style: str = "primary", subtitle: str = None):
    """Display a themed section header that adapts to current theme.

    Args:
        title: The section title (can include emoji)
        style: Color style - 'primary', 'success', 'warning', 'info', 'error', or 'accent'
        subtitle: Optional subtitle text
    """
    theme = get_current_theme()

    # Validate style parameter
    if style not in _VALID_STYLES:
        style = "primary"

    # Map style to theme colors
    style_colors = {
        "primary": (theme.primary, theme.bg_card),
        "success": (theme.success, theme.bg_card),
        "warning": (theme.warning, theme.bg_card),
        "info": (theme.info, theme.bg_card),
        "error": (theme.error, theme.bg_card),
        "accent": (theme.accent, theme.bg_card),
    }

    text_color, bg_color = style_colors[style]

    # Escape HTML in title and subtitle to prevent injection
    safe_title = _escape_html(title)
    safe_subtitle = _escape_html(subtitle) if subtitle else None

    subtitle_html = (
        f'<p style="margin: 8px 0 0 0; color: {theme.text_muted}; '
        f'font-size: 0.9rem; font-family: var(--theme-font-mono);">{safe_subtitle}</p>'
        if safe_subtitle else ''
    )

    st.markdown(f"""
    <div style="padding: 16px; background: {bg_color}; border-radius: 12px; margin-bottom: 20px; border-left: 4px solid {text_color};">
        <h3 style="margin: 0; color: {text_color}; font-family: var(--theme-font-mono);">{safe_title}</h3>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def themed_banner_html(title: str, message: str) -> str:
    """Generate HTML for a themed banner."""
    theme = get_current_theme()
    return f"""
    <div class="themed-banner">
        <h3 style="margin: 0; color: white; font-family: var(--theme-font-mono);">{title}</h3>
        <p style="margin: 10px 0 0 0; opacity: 0.9; color: white; font-family: var(--theme-font-mono);">
            {message}
        </p>
    </div>
    """


# Export theme utilities
__all__ = [
    'inject_custom_css',
    'animated_metric',
    'feature_card',
    'hero_section',
    'gradient_banner',
    'animated_progress_bar',
    'score_bar',
    'rank_badge',
    'animated_list_item',
    'loading_shimmer',
    'display_lottie',
    'glass_card',
    'vectory_logo',
    'animated_logo',
    'sidebar_logo',
    'app_footer',
    'page_header',
    'terminal_output',
    'code_block',
    'status_badge',
    'dev_stat',
    'themed_header',
    'themed_card',
    'themed_banner_html',
    'section_header',
    'get_current_theme',
    'hex_to_rgb',
    'LOTTIE_URLS',
]


# Public alias for hex_to_rgb
hex_to_rgb = _hex_to_rgb

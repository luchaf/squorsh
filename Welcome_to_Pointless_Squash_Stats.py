import streamlit as st
from color_palette import PRIMARY, SECONDARY, TERTIARY

# App title
st.title('Welcome to "Pointless Squash Stats"!')
st.write("Where squash isn't just a vegetable.")

# Intro content
st.write(
    """
In the realm of squash, where every swing, smash, and slide counts, it's often the banter that lingers longer than the actual scores. But why rely solely on memory and wit? Memories fade, your digital footprints won't. Back your pre- and postgame trashtalk with serious, objective performance statistics.
""",
    color=TERTIARY,
)

# Section 1
st.subheader("1. Racquet Records: Document your match results")
st.write(
    """
Document each exhilarating (or ego-crushing) match result. Remember, you'll need your seed later to relive (or regret) those game stats.
""",
    color=SECONDARY,
)
st.write(
    """
Step into our digital hall of fame (and occasional shame). Each match you play, every score you chalk up, is more than just a number—it's a testament to skill, strategy, and that sneaky drop shot you've been perfecting. Whether it's an earth-shattering victory or a humble hiccup, it'll be archived here in all its glory.
""",
    color=TERTIARY,
)

# Section 2
st.subheader("2. Overanalysis Oasis: Delving Deep into Data Details.")
st.write(
    """
Data has a story to tell, and boy, do we love to narrate! From the trajectory of your winning streaks to the pattern of your losses, we dissect every detail. Want to know how often you've beaten Chris on a Wednesday evening? Or which month you truly peaked? We're on it, with charts, graphs, and narratives that are almost obsessively granular.
""",
    color=SECONDARY,
)

# Footer
st.write(
    """
Step into "Pointless Squash Stats", where numbers narrate the nonsense and amusement is always ace.
""",
    color=TERTIARY,
)
st.write(
    "👈 Explore by clicking on the chevron on the top of your screen to unfold the sidebar!!",
    color=PRIMARY,
)

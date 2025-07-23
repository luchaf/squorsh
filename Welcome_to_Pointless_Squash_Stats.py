import streamlit as st
from color_palette import PRIMARY, SECONDARY, TERTIARY

# App title
st.title('🏓 Willkommen bei "Pointless Squash Stats"!')
st.write("Wo Squash mehr als nur ein Kürbis ist... und Statistiken wichtiger werden als das Spiel selbst! 🎾")

# Intro content
st.markdown(
    """
    ## 🎯 **Das ultimative Turnier für alle Spielstärken!**
    
    Stellt euch vor: Ein Turnier, wo der Profi mit Augenklappe und Holzschläger antritt, 
    während der Anfänger völlig normal spielen darf. Klingt absurd? Ist es auch! 
    Aber genau das macht es so wunderbar pointless - und unterhaltsam! 🤪
    """,
    unsafe_allow_html=True
)

st.divider()

# How it works
st.subheader("🗺️ Dein Weg zum Ruhm:")


st.markdown(
    """
    ### 📝 **1. Namen eintragen**
    Links oben findest du das Menü. Zuerst musst du dich anmelden! Geh zur **"Pointless Sign Up"** Seite 
    und trag deinen Namen ein. Ohne Namen kein Ruhm! 🏆
    
    ### ⚔️ **2. Herausfordern & Kämpfen**
    Fordere JEDEN heraus! Egal ob Profi oder Anfänger - hier kann jeder 
    gegen jeden antreten. Ein Satz bis 15 Punkte entscheidet!
    
    ### 📊 **3. Ergebnisse eintragen**
    Nach dem Match: Trage deine Ergebnisse in der Seite "Pointless Racquet Records" ein. Aus deinen Ergebnissen wird im Tab "Ranking" ein Score erstellt. Die Person mit dem höchsten Score am Ende des Tages gewinnt. 🤓
    
    **Regel:** Du kannst maximal 2x gegen dieselbe Person spielen. 
    Fordere so viele verschiedene Leute wie möglich heraus!
    """
)



st.divider()

# Handicap system
st.subheader("🎪 Das Handicap-System")
st.markdown(
    """
    Falls du dachtest das wars schon, hast du dich getäuscht! Je besser du bist, desto größer dein Handicap. Dabei gibt es zwei Kategorien:
    
    **🪵 Kategorie 1 Handicaps:**
    - Alter Holzschläger 🏏
    - Augenklappe 🏴‍☠️
    - Schwarz gepunktete Brille 🕶️
    - Gewichtsweste
    
    **🏊 Kategorie 2 Handicaps:**
    - 4 Schwimmflügel! 🏊‍♂️
    - Prisma-Brille 🌈
    - Farbbrille 🔴
    """
)
st.markdown(
    """
    Je größer der Unterschied zwischen den Spielern, desto mehr Handicaps bekommt der Bessere:
    
    | Ranking-Differenz | Handicap für den Besseren |
    |-------------------|---------------------------|
    | 2-4 Punkte | 1x Kategorie 1 (z.B. Holzschläger) |
    | 4-6 Punkte | 1x Kategorie 2 (z.B. Schwimmflügel) |
    | 6-8 Punkte | 1x Kategorie 1 + 1x Kategorie 2 |
    | 8+ Punkte | 2x Kategorie 2 (Viel Spaß! 😈) |
    
    
    """
)

st.divider()

# Footer
st.markdown(
    """
    ## 🎉 **Du kannst nicht genug bekommen?**
    
    Wir wären nicht "Pointless Squash Stats" - wenn wir dich nicht mit mehr überflüssigen Statisiken versorgen würden, als du dir ansehen kannst. 
    Auf der Seite "Pointless Overanalysis Oasis" kannst du deine Performance bis ins kleinste Detail analysieren.
    
    **Los geht's!** 👈 Klick auf den Pfeil oben links, um die Sidebar zu öffnen 
    und dein Abenteuer zu beginnen!
    """,
    unsafe_allow_html=True
)

st.balloons()

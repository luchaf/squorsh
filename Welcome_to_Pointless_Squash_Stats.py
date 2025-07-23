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
    Aber genau das macht es so wunderbar fair - und unterhaltsam! 🤪
    """,
    unsafe_allow_html=True
)

st.divider()

# How it works
st.subheader("🎮 So funktioniert das System:")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(
        """
        ### 📝 **1. Namen eintragen**
        Zuerst musst du dich anmelden! Geh zur **"Pointless Sign Up"** Seite 
        und trag deinen Namen ein. Ohne Namen kein Ruhm! 🏆
        
        ### ⚔️ **2. Herausfordern & Kämpfen**
        Fordere JEDEN heraus! Egal ob Profi oder Anfänger - hier kann jeder 
        gegen jeden antreten. Ein Satz bis 15 Punkte entscheidet!
        
        ### 📊 **3. Ergebnisse eintragen**
        Nach dem Match: Ergebnis in die App! Unser geheimer Algorithmus 
        berechnet dann dein Ranking mit einer völlig überdrehten Formel! 🤓
        """
    )

with col2:
    st.markdown(
        """
        ### 🎭 **4. Handicaps = Fair Play!**
        Je besser du bist, desto absurder wird's:
        
        **🪵 Kategorie 1 Handicaps:**
        - Alter Holzschläger 🏏
        - Augenklappe 🏴‍☠️
        - Schwarz gepunktete Brille 🕶️
        - Gewichte am Bein ⚖️
        
        **🏊 Kategorie 2 Handicaps:**
        - 4 Schwimmflügel! 🏊‍♂️
        - Prisma-Brille 🌈
        - Farbbrille 🔴
        - Gewichte am Arm 💪
        """
    )

st.divider()

# Handicap system
st.subheader("🎪 Das Handicap-System")

st.markdown(
    """
    Unser Ranking-System macht Fairness zu einer Wissenschaft für sich! 
    Je größer der Unterschied zwischen den Spielern, desto mehr Handicaps bekommt der Bessere:
    
    | Ranking-Differenz | Handicap für den Besseren |
    |-------------------|---------------------------|
    | 2-4 Punkte | 1x Kategorie 1 (z.B. Holzschläger) |
    | 4-6 Punkte | 1x Kategorie 2 (z.B. Schwimmflügel) |
    | 6-8 Punkte | 1x Kategorie 1 + 1x Kategorie 2 |
    | 8+ Punkte | 2x Kategorie 2 (Viel Spaß! 😈) |
    
    **Regel:** Du kannst maximal 2x gegen dieselbe Person spielen. 
    Fordere so viele verschiedene Leute wie möglich heraus!
    """
)

st.divider()

# Navigation guide
st.subheader("🗺️ Dein Weg zum Ruhm:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        ### 📝 **Sign Up**
        Hier trägst du deinen Namen ein 
        und siehst alle anderen Teilnehmer.
        """
    )

with col2:
    st.markdown(
        """
        ### 🏓 **Racquet Records**
        Hier trägst du Match-Ergebnisse ein 
        und siehst alle bisherigen Spiele.
        """
    )

with col3:
    st.markdown(
        """
        ### 🏆 **Rankings**
        Hier siehst du die aktuelle Rangliste 
        mit unserer geheimen Super-Formel!
        """
    )

st.divider()

# Footer
st.markdown(
    """
    ## 🎉 **Bereit für den Spaß?**
    
    Willkommen bei "Pointless Squash Stats" - wo Statistiken wichtiger werden als das Spiel, 
    Handicaps wunderbar absurd und der Spaß garantiert ist! 
    
    **Los geht's!** 👈 Klick auf den Pfeil oben links, um die Sidebar zu öffnen 
    und dein Abenteuer zu beginnen!
    """,
    unsafe_allow_html=True
)

st.balloons()

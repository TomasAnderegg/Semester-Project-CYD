import duckdb
import pandas as pd

# =============================================================================
# 🚀 EXPLORATION DE TOUTES LES ENTITÉS DANS LA BASE CRUNCHBASE
# =============================================================================
print("=" * 80)
print("🧠 EXPLORATION DE LA BASE CRUNCHBASE : LISTE ET HEADS DES ENTITÉS")
print("=" * 80)

# 📁 Chemin vers ta base DuckDB
db_path = r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Crunchbase dataset\crunchbase.duckdb"

# Connexion
con = duckdb.connect(db_path)

# =============================================================================
# 📋 LISTER TOUTES LES TABLES DISPONIBLES
# =============================================================================
print("\n📊 Récupération de la liste des entités disponibles...")
tables_query = """
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'main'
ORDER BY table_name
"""

tables_df = con.execute(tables_query).fetchdf()
tables = tables_df["table_name"].tolist()

print(f"✅ {len(tables)} tables trouvées dans le schéma 'main' :\n")
for i, t in enumerate(tables, start=1):
    print(f"   {i:2d}. {t}")

# =============================================================================
# 🔍 EXPLORATION DE CHAQUE TABLE
# =============================================================================
print("\n" + "=" * 80)
print("🔎 APERÇU DES ENTITÉS (head des tables)")
print("=" * 80)

for idx, table in enumerate(tables, start=1):
    print(f"\n{'-' * 80}")
    print(f"📁 [{idx}/{len(tables)}] Table: {table}")
    print("-" * 80)
    try:
        # Compter les lignes
        count_query = f"SELECT COUNT(*) AS count FROM main.{table}"
        count = con.execute(count_query).fetchdf()["count"].iloc[0]
        print(f"📦 Nombre de lignes : {count:,}")

        # Charger les 5 premières lignes
        preview_query = f"SELECT * FROM main.{table} LIMIT 5"
        df = con.execute(preview_query).fetchdf()

        # Afficher la liste des colonnes
        print(f"🧩 Colonnes ({len(df.columns)}): {', '.join(df.columns)}")

        # Afficher les 5 premières lignes
        print("\n🧾 Aperçu des données :")
        print(df.head())

    except Exception as e:
        print(f"⚠️ Erreur lors de l'exploration de {table}: {str(e)[:200]}")

# =============================================================================
# 🔚 FERMETURE
# =============================================================================
con.close()
print("\n" + "=" * 80)
print("✅ EXPLORATION TERMINÉE AVEC SUCCÈS")
print("=" * 80)

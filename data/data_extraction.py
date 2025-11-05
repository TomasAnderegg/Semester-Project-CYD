import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime

# =============================================================================
# 🚀 EXPORT DE LA TABLE 'organization' AVEC SEULEMENT QUELQUES COLONNES
# =============================================================================
print("=" * 80)
print("🧠 EXPORT DE LA TABLE 'organization' DE CRUNCHBASE (colonnes sélectionnées)")
print("=" * 80)

# 📁 Chemins
db_path = r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Crunchbase dataset\crunchbase.duckdb"
output_dir = Path("savings/csv/entities_exploration")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\n📂 Dossier de sortie créé : {output_dir}")

# Connexion
con = duckdb.connect(db_path)

# Colonnes à extraire
columns_to_extract = ["uuid", "name", "category_groups_list","category_list", "short_description"]
table_name = "organizations"

try:
    # Vérifier que la table existe
    tables_df = con.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'main'
    """).fetchdf()

    if table_name not in tables_df['table_name'].tolist():
        raise ValueError(f"Table '{table_name}' introuvable dans la base.")

    # Construire la requête SQL pour ne prendre que certaines colonnes
    cols_sql = ", ".join(columns_to_extract)
    df = con.execute(f"SELECT {cols_sql} FROM main.{table_name}").fetchdf()

    # Afficher un aperçu
    print(f"\n🧩 Colonnes ({len(df.columns)}): {', '.join(df.columns)}")
    print("\n🧾 Aperçu des 5 premières lignes :")
    print(df.head())

    # Sauvegarder en CSV
    csv_path = output_dir / f"{table_name}_selected_columns.csv"
    df.to_csv(csv_path, index=False)
    file_size_mb = csv_path.stat().st_size / (1024 ** 2)
    print(f"\n💾 CSV sauvegardé : {csv_path} ({file_size_mb:.2f} MB)")

except Exception as e:
    print(f"⚠️ Erreur lors de l'export de '{table_name}': {e}")

# Créer un README minimal
readme_path = output_dir / "README.txt"
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("EXPORT DE LA TABLE 'organization' DE CRUNCHBASE (colonnes sélectionnées)\n")
    f.write("="*80 + "\n\n")
    f.write(f"Date d'export : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Base de données source : {db_path}\n\n")
    f.write(f"Table exportée : {table_name}\n")
    f.write(f"Colonnes : {', '.join(columns_to_extract)}\n")
    f.write(f"Nombre de lignes : {len(df):,}\n")
    f.write(f"Taille CSV : {file_size_mb:.2f} MB\n")
print(f"📝 README créé : {readme_path}")

# Fermeture de la connexion
con.close()
print("\n✅ EXPORT TERMINÉ AVEC SUCCÈS")
print(f"📂 Fichier CSV disponible dans : {output_dir.absolute()}")

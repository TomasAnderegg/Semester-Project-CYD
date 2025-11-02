import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime

# =============================================================================
# 🚀 EXPLORATION ET EXPORT DE TOUTES LES ENTITÉS DANS LA BASE CRUNCHBASE
# =============================================================================
print("=" * 80)
print("🧠 EXPLORATION ET EXPORT DE LA BASE CRUNCHBASE")
print("=" * 80)

# 📁 Chemins
db_path = r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Crunchbase dataset\crunchbase.duckdb"
output_dir = Path("savings/csv/entities_exploration")

# Créer le dossier de sortie
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\n📂 Dossier de sortie créé : {output_dir}")

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
# 🔍 EXPLORATION ET EXPORT DE CHAQUE TABLE
# =============================================================================
print("\n" + "=" * 80)
print("🔎 EXPLORATION ET EXPORT DES ENTITÉS")
print("=" * 80)

# Statistiques globales
total_rows = 0
total_size_mb = 0
export_summary = []

for idx, table in enumerate(tables, start=1):
    print(f"\n{'-' * 80}")
    print(f"📁 [{idx}/{len(tables)}] Table: {table}")
    print("-" * 80)
    
    try:
        # Compter les lignes
        count_query = f"SELECT COUNT(*) AS count FROM main.{table}"
        count = con.execute(count_query).fetchdf()["count"].iloc[0]
        print(f"📦 Nombre de lignes : {count:,}")
        total_rows += count
        
        # Charger TOUTES les données (attention à la mémoire!)
        full_query = f"SELECT * FROM main.{table}"
        df = con.execute(full_query).fetchdf()
        
        # Afficher la liste des colonnes
        print(f"🧩 Colonnes ({len(df.columns)}): {', '.join(df.columns[:10])}" + 
              (f", ... et {len(df.columns) - 10} autres" if len(df.columns) > 10 else ""))
        
        # Afficher les 5 premières lignes
        print("\n🧾 Aperçu des données (5 premières lignes) :")
        print(df.head())
        
        # Sauvegarder en CSV
        csv_filename = f"{table}.csv"
        csv_path = output_dir / csv_filename
        
        print(f"\n💾 Sauvegarde en CSV...")
        df.to_csv(csv_path, index=False)
        
        # Calculer la taille du fichier
        file_size_mb = csv_path.stat().st_size / (1024 ** 2)
        total_size_mb += file_size_mb
        
        print(f"✅ Sauvegardé : {csv_path}")
        print(f"   Taille : {file_size_mb:.2f} MB")
        print(f"   Shape : {df.shape}")
        
        # Ajouter au résumé
        export_summary.append({
            'table': table,
            'rows': count,
            'columns': len(df.columns),
            'size_mb': file_size_mb,
            'filename': csv_filename
        })
        
    except Exception as e:
        print(f"⚠️ Erreur lors de l'export de {table}: {str(e)[:200]}")
        export_summary.append({
            'table': table,
            'rows': 0,
            'columns': 0,
            'size_mb': 0,
            'filename': f"{table}.csv",
            'error': str(e)[:100]
        })

# =============================================================================
# 📊 RÉSUMÉ FINAL
# =============================================================================
print("\n" + "=" * 80)
print("📊 RÉSUMÉ DE L'EXPORT")
print("=" * 80)

# Créer un DataFrame de résumé
summary_df = pd.DataFrame(export_summary)

print(f"\n📈 Statistiques globales :")
print(f"   Total de lignes exportées : {total_rows:,}")
print(f"   Taille totale des CSV : {total_size_mb:.2f} MB")
print(f"   Nombre de tables exportées : {len([s for s in export_summary if 'error' not in s])}/{len(tables)}")

print(f"\n📋 Détails par table :")
print(summary_df.to_string(index=False))

# Sauvegarder le résumé
summary_path = output_dir / "export_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\n💾 Résumé sauvegardé : {summary_path}")

# Créer un fichier README
readme_path = output_dir / "README.txt"
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("EXPORT DES ENTITÉS CRUNCHBASE\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Date d'export : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Base de données source : {db_path}\n\n")
    f.write(f"Nombre de tables exportées : {len(tables)}\n")
    f.write(f"Total de lignes : {total_rows:,}\n")
    f.write(f"Taille totale : {total_size_mb:.2f} MB\n\n")
    f.write("=" * 80 + "\n")
    f.write("LISTE DES FICHIERS\n")
    f.write("=" * 80 + "\n\n")
    
    for item in export_summary:
        f.write(f"📄 {item['filename']}\n")
        f.write(f"   Lignes : {item['rows']:,}\n")
        f.write(f"   Colonnes : {item['columns']}\n")
        f.write(f"   Taille : {item['size_mb']:.2f} MB\n")
        if 'error' in item:
            f.write(f"   ⚠️ Erreur : {item['error']}\n")
        f.write("\n")

print(f"📝 README créé : {readme_path}")

# =============================================================================
# 🗂️ STRUCTURE DES FICHIERS
# =============================================================================
print("\n" + "=" * 80)
print("🗂️  STRUCTURE DES FICHIERS CRÉÉS")
print("=" * 80)
print(f"\n📁 {output_dir}/")
print(f"   ├── export_summary.csv")
print(f"   ├── README.txt")

for item in sorted(export_summary, key=lambda x: x['table']):
    status = "✅" if 'error' not in item else "❌"
    print(f"   ├── {status} {item['filename']} ({item['size_mb']:.2f} MB)")

# =============================================================================
# 🔚 FERMETURE
# =============================================================================
con.close()
print("\n" + "=" * 80)
print("✅ EXPLORATION ET EXPORT TERMINÉS AVEC SUCCÈS")
print(f"📂 Tous les fichiers sont dans : {output_dir.absolute()}")
print("=" * 80)

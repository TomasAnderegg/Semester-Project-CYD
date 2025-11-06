import duckdb
import pandas as pd
from pathlib import Path

# 📁 Chemin vers la base de données
db_path = r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Crunchbase dataset\crunchbase.duckdb"
output_dir = Path("savings/csv/funding_rounds")
output_dir.mkdir(parents=True, exist_ok=True)

# Nom de l'entreprise
company_name = "Meta"

# Connexion
con = duckdb.connect(db_path)

try:
    # 1️⃣ Vérifier que l'entreprise existe et récupérer son UUID
    query_uuid = f"""
        SELECT uuid, name
        FROM main.organizations
        WHERE name = '{company_name.replace("'", "''")}'
    """
    df_company = con.execute(query_uuid).fetchdf()

    if df_company.empty:
        print(f"⚠️ L'entreprise '{company_name}' n'existe pas dans organizations.")
    else:
        company_uuid = df_company['uuid'].iloc[0]
        print(f"✅ L'entreprise '{company_name}' existe avec UUID : {company_uuid}")

        # 2️⃣ Chercher les levées de fonds dans funding_rounds
        query_funding = f"""
            SELECT announced_on
            FROM main.funding_rounds
            WHERE org_uuid = '{company_uuid}'
            ORDER BY announced_on
        """
        df_funding = con.execute(query_funding).fetchdf()

        if df_funding.empty:
            print(f"⚠️ Aucune levée de fonds trouvée pour '{company_name}'.")
        else:
            print(f"\n🧾 Levées de fonds trouvées ({len(df_funding)} lignes) :")
            print(df_funding.head(20))  # aperçu des 20 premières levées

            # Sauvegarder CSV
            csv_path = output_dir / f"funding_rounds_dates_{company_name.replace(' ', '_')}.csv"
            if csv_path.exists():
                csv_path.unlink()
            df_funding.to_csv(csv_path, index=False)
            print(f"\n💾 CSV sauvegardé : {csv_path}")

except Exception as e:
    print(f"⚠️ Erreur : {e}")

finally:
    con.close()
    print("\n✅ Extraction terminée.")

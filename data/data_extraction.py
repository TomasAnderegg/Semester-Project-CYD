import duckdb
import pandas as pd
from pathlib import Path

# 📁 Chemins
db_path = r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Crunchbase dataset\crunchbase.duckdb"
comp_tech_csv = r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Code\savings\csv_results\companies_rank_500_default.csv"
output_dir = Path(r"C:\Users\tjga9\Documents\Tomas\EPFL\MA3\CYD PDS\Code\savings\csv/funding_rounds")
output_dir.mkdir(parents=True, exist_ok=True)

# Charger les entreprises
df_companies = pd.read_csv(comp_tech_csv)

# Connexion à DuckDB
con = duckdb.connect(db_path)

MINIMUM_NUM_FUNDING = 5

# Liste pour stocker toutes les levées de fonds
all_funding_data = []

for company_name in df_companies["final_configuration"]:
    try:
        # 1️⃣ Vérifier que l'entreprise existe et récupérer son UUID
        query_uuid = f"""
            SELECT uuid, name
            FROM main.organizations
            WHERE name = '{company_name.replace("'", "''")}'
        """
        df_company = con.execute(query_uuid).fetchdf()

        if df_company.empty:
            print(f"L'entreprise '{company_name}' n'existe pas dans organizations.")
            continue

        company_uuid = df_company['uuid'].iloc[0]
        print(f"'{company_name}' existe avec UUID : {company_uuid}")

        # 2️⃣ Récupérer les levées de fonds
        query_funding = f"""
            SELECT '{company_name}' AS company_name, announced_on
            FROM main.funding_rounds
            WHERE org_uuid = '{company_uuid}'
            ORDER BY announced_on
        """
        df_funding = con.execute(query_funding).fetchdf()

        if df_funding.empty:
            print(f" Aucune levée de fonds trouvée pour '{company_name}'.")
        else:
            print(f" {len(df_funding)} levées de fonds trouvées pour '{company_name}'.")
            all_funding_data.append(df_funding)

    except Exception as e:
        print(f"⚠️ Erreur pour '{company_name}' : {e}")

# Concaténer toutes les données et sauvegarder le CSV global
if all_funding_data:
    df_all_funding = pd.concat(all_funding_data, ignore_index=True)
    csv_all_path = output_dir / "all_companies_funding_rounds.csv"
    df_all_funding.to_csv(csv_all_path, index=False)
    print(f"\n💾 CSV global sauvegardé : {csv_all_path}")

    # 🔹 Filtrer entreprises avec >=5 levées de fonds
    df_counts = df_all_funding.groupby("company_name").size().reset_index(name="announced_on")
    df_5plus = df_counts[df_counts["announced_on"] >= MINIMUM_NUM_FUNDING].sort_values(by="announced_on", ascending=False)

    csv_5plus_path = output_dir / "companies_5plus_funding.csv"
    df_5plus.to_csv(csv_5plus_path, index=False)
    print(f"💾 CSV des entreprises avec >=5 levées de fonds sauvegardé : {csv_5plus_path}")

else:
    print("⚠️ Aucune levée de fonds trouvée pour toutes les entreprises.")

# Fermer la connexion
con.close()
print("\n✅ Extraction terminée.")

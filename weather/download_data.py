from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import requests


def download_and_extract_zip(url):
    response = requests.get(url)
    if response.status_code == 200:
        with ZipFile(BytesIO(response.content)) as the_zip:
            for zip_info in the_zip.infolist():
                with the_zip.open(zip_info) as file:
                    df = pd.read_csv(file)
                    yield df
    else:
        print(f"Failed to download data from {url}")


def main():
    base_url = "https://www.bgc-jena.mpg.de/wetter/"
    periods = [
                  "mpi_saale",
                  "mpi_saale_2023b", "mpi_saale_2023a",
                  "mpi_saale_2022b", "mpi_saale_2022a",
                  "mpi_saale_2021b", "mpi_saale_2021a",
                  "mpi_saale_2020b", "mpi_saale_2020a",
                  "mpi_saale_2019b", "mpi_saale_2019a",
                  "mpi_saale_2018b", "mpi_saale_2018a",
                  "mpi_saale_2017b", "mpi_saale_2017a",
                  "mpi_saale_2016b", "mpi_saale_2016a",
                  "mpi_saale_2015b", "mpi_saale_2015a",
                  "mpi_saale_2014b", "mpi_saale_2014a",
                  "mpi_saale_2013b", "mpi_saale_2013a",
              ][::-1]  # Reversed to append chronologically

    all_data = pd.DataFrame()

    for period in periods:
        url = f"{base_url}{period}.zip"
        print(url)
        for df in download_and_extract_zip(url):
            all_data = pd.concat([all_data, df], ignore_index=True)

    all_data.to_csv("data/weather_data.csv", index=False)
    print("Data has been compiled and saved to combined_weather_data.csv.")


if __name__ == "__main__":
    main()

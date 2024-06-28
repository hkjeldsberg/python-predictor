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
    start = 2009
    stop = 2017
    periods = []
    for year in range(start, stop):
        for n in ["a", "b"]:
            periods.append(f"mpi_saale_{year}{n}")

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

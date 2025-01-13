import torch
from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
)

config = SHConfig()
config.sh_client_id = #client_id
config.sh_client_secret = #client_secret
config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'

def request(init_date, final_date, coord, size):
    """
    Sends a request to SentinelHub for satellite data (Sentinel-2).

    This function generates a request for downloading Sentinel-2 data from the SentinelHub service
    based on the specified date range, geographic coordinates, and image size. The function constructs
    an evaluation script to request all Sentinel-2 bands (B01 to B12) at the specified location and time period.

    Parameters
    ----------
    init_date : str
        The start date of the time period for the satellite data, formatted as 'YYYY-MM-DD'.
    final_date : str
        The end date of the time period for the satellite data, formatted as 'YYYY-MM-DD'.
    coord : tuple
        A tuple (min_longitude, min_latitude, max_longitude, max_latitude) defining the bounding box
        of the area of interest in WGS84 coordinates.
    size : tuple
        A tuple (width, height) specifying the resolution or size of the output image.

    Returns
    -------
    numpy.ndarray
        A NumPy array containing the downloaded satellite data for the specified time period and region,
        with the shape (num_dates, 13, height, width), where 13 corresponds to the 13 bands.
    """
    evalscript_all_bands = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"],
                    units: "REFLECTANCE"
                }],
                output: {
                    bands: 13,
                    sampleType: SampleType.FLOAT32
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B01,
                    sample.B02,
                    sample.B03,
                    sample.B04,
                    sample.B05,
                    sample.B06,
                    sample.B07,
                    sample.B08,
                    sample.B8A,
                    sample.B09,
                    sample.B10,
                    sample.B11,
                    sample.B12];
        }
    """
    bbox = BBox(bbox=coord, crs=CRS.WGS84)
    request_all_bands = SentinelHubRequest(
        evalscript=evalscript_all_bands,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C.define_from(
                    "s2l1c", service_url=config.sh_base_url
                ),
                time_interval=(init_date, final_date),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    return request_all_bands.get_data()

def data(dates, coord, size):
    """
    Downloads satellite data for a range of dates and a specific region.

    This function processes a list of date pairs, where each pair represents a start and end date for
    which satellite data is requested. It combines the data from each date into a single tensor and
    excludes the 10th band (which represents the cloud mask).

    Parameters
    ----------
    dates : list of tuples
        A list of date pairs, each containing a start date and an end date formatted as ('YYYY-MM-DD', 'YYYY-MM-DD').
    coord : tuple
        A tuple (min_longitude, min_latitude, max_longitude, max_latitude) defining the bounding box
        of the area of interest in WGS84 coordinates.
    size : tuple
        A tuple (width, height) specifying the resolution or size of the output image.

    Returns
    -------
    torch.Tensor
        A tensor containing the downloaded data for each date, with the shape
        (num_dates, 12, height, width), excluding the 10th band (cloud mask).
    """
    aux = []
    for date in dates:
        initial = date[0]
        final = date[1]
        aux.append(torch.tensor(request(initial, final, coord, size)))
    tensor = torch.cat(aux, dim=0).permute(0, 3, 1, 2)
    index = [i for i in range(tensor.size(1)) if i != 10]  # Exclude the 10th band (cloud mask)
    return tensor[:, index, :, :]
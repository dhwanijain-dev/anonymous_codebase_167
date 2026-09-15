from PIL import Image
import numpy as np


ALLOWED_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
}


def validate_extension(filename: str):

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported format: {extension}. "
                f"Supported formats: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )


def load_image_bytes(
    data: bytes,
    filename: str,
) -> Image.Image:

    extension = Path(filename).suffix.lower()

    # --------------------------------------------------------
    # Normal image
    # --------------------------------------------------------

    if extension in {
        ".png",
        ".jpg",
        ".jpeg",
    }:

        image = Image.open(
            io.BytesIO(data)
        )

        return image.convert("RGB")


    # --------------------------------------------------------
    # GeoTIFF
    # --------------------------------------------------------

    if extension in {
        ".tif",
        ".tiff",
    }:

        try:

            import rasterio

            with rasterio.MemoryFile(data) as memfile:

                with memfile.open() as dataset:

                    array = dataset.read()

            # ------------------------------------------------
            # Convert multispectral / SAR to display image
            # ------------------------------------------------

            if array.ndim != 3:

                raise ValueError(
                    "Expected raster with dimensions "
                    "(bands, height, width)."
                )

            bands = array.shape[0]

            # -----------------------------------------------
            # Single band → grayscale
            # -----------------------------------------------

            if bands == 1:

                channel = array[0]

                image = normalize_band(
                    channel
                )

                return Image.fromarray(
                    image,
                    mode="L"
                ).convert("RGB")


            # -----------------------------------------------
            # 3+ bands → first three
            # -----------------------------------------------

            if bands >= 3:

                rgb = np.stack(
                    [
                        array[0],
                        array[1],
                        array[2],
                    ],
                    axis=-1,
                )

                rgb = normalize_rgb(
                    rgb
                )

                return Image.fromarray(
                    rgb,
                    mode="RGB"
                )

        except Exception as exc:

            raise HTTPException(
                status_code=400,
                detail=f"Could not read GeoTIFF: {exc}",
            )

    raise HTTPException(
        status_code=400,
        detail="Unable to decode image.",
    )


def normalize_band(
    band: np.ndarray,
) -> np.ndarray:

    band = band.astype(np.float32)

    valid = np.isfinite(band)

    if not valid.any():

        return np.zeros(
            band.shape,
            dtype=np.uint8,
        )

    low, high = np.percentile(
        band[valid],
        [2, 98],
    )

    if high <= low:

        high = low + 1e-6

    result = (
        (band - low)
        / (high - low)
    )

    result = np.clip(
        result,
        0,
        1,
    )

    return (
        result * 255
    ).astype(np.uint8)


def normalize_rgb(
    rgb: np.ndarray,
) -> np.ndarray:

    output = np.zeros_like(
        rgb,
        dtype=np.uint8,
    )

    for channel in range(3):

        output[:, :, channel] = normalize_band(
            rgb[:, :, channel]
        )

    return output

def create_pair_image(
    image1: Image.Image,
    image2: Image.Image,
) -> Image.Image:

    # Same size
    image1 = image1.resize(
        (512, 512)
    )

    image2 = image2.resize(
        (512, 512)
    )

    canvas = Image.new(
        "RGB",
        (
            1024,
            512,
        ),
    )

    canvas.paste(
        image1,
        (0, 0),
    )

    canvas.paste(
        image2,
        (512, 0),
    )

    return canvas

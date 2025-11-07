#!/bin/bash
#
# Requires bash version >= 4
# 
# The script uses the command line tool 'curl' for querying
# the MAST Download service for public and protected data. 
#

type -P curl >&/dev/null || { echo "This script requires curl. Exiting." >&2; exit 1; }



# Download Product Files:



cat <<EOT
<<< Downloading File: mast:JWST/product/jw09235001001_03101_00001-seg005_mirimage_uncal.fits
                  To: jw09235001001_03101_00001-seg005_mirimage_uncal.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output './jw09235001001_03101_00001-seg005_mirimage_uncal.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2025-10-28T1610.sh&uri=mast:JWST/product/jw09235001001_03101_00001-seg005_mirimage_uncal.fits'





cat <<EOT
<<< Downloading File: mast:JWST/product/jw09235001001_03101_00001-seg003_mirimage_uncal.fits
                  To: jw09235001001_03101_00001-seg003_mirimage_uncal.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output './jw09235001001_03101_00001-seg003_mirimage_uncal.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2025-10-28T1610.sh&uri=mast:JWST/product/jw09235001001_03101_00001-seg003_mirimage_uncal.fits'





cat <<EOT
<<< Downloading File: mast:JWST/product/jw09235001001_03101_00001-seg001_mirimage_uncal.fits
                  To: jw09235001001_03101_00001-seg001_mirimage_uncal.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output './jw09235001001_03101_00001-seg001_mirimage_uncal.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2025-10-28T1610.sh&uri=mast:JWST/product/jw09235001001_03101_00001-seg001_mirimage_uncal.fits'





cat <<EOT
<<< Downloading File: mast:JWST/product/jw09235001001_03101_00001-seg002_mirimage_uncal.fits
                  To: ${DOWNLOAD_FOLDER}/JWST/jw09235001001_03101_00001-seg002_mirimage/jw09235001001_03101_00001-seg002_mirimage_uncal.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output './jw09235001001_03101_00001-seg002_mirimage_uncal.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2025-10-28T1610.sh&uri=mast:JWST/product/jw09235001001_03101_00001-seg002_mirimage_uncal.fits'





cat <<EOT
<<< Downloading File: mast:JWST/product/jw09235001001_03101_00001-seg004_mirimage_uncal.fits
                  To: ${DOWNLOAD_FOLDER}/JWST/jw09235001001_03101_00001-seg004_mirimage/jw09235001001_03101_00001-seg004_mirimage_uncal.fits
EOT

curl --globoff --location-trusted -f --progress-bar --create-dirs $CONT --output './jw09235001001_03101_00001-seg004_mirimage_uncal.fits' 'https://mast.stsci.edu/api/v0.1/Download/file?bundle_name=MAST_2025-10-28T1610.sh&uri=mast:JWST/product/jw09235001001_03101_00001-seg004_mirimage_uncal.fits'




exit 0

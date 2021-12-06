from torch.hub import download_url_to_file

openmicurl = "https://zenodo.org/record/1432913/files/openmic-2018-v1.0.0.tgz?download=1"
download_target = "openmic-2018-v1.0.0.tgz"

print("Downloading OpenMIC from zenodo")
download_url_to_file(openmicurl, download_target)

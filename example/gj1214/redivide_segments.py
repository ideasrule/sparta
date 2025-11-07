import astropy.io.fits
import matplotlib.pyplot as plt
import numpy as np
import copy

#Integration numbers, counting from beginning
transit_begin = 10824
transit_end = 11302
seg5_begin = 10528
seg6_begin = 11000

def write_pre_transit_segment(input_filename="old_uncalibrated/jw01803001001_04103_00003-seg005_mirimage_uncal.fits", output_filename="jw01803001001_04103_00003-seg005a_mirimage_uncal.fits"):
    seg1 = astropy.io.fits.open(input_filename)
    seg1_header = dict(seg1[0].header)
    times = np.linspace(seg1[0].header["BSTRTIME"], seg1[0].header["BENDTIME"], seg1[1].data.shape[0])
    cutoff = transit_begin - seg5_begin
    transit_intstart = seg1[0].header["INTSTART"] + cutoff
    transit_data = np.copy(seg1[1].data[cutoff:])

    header_hdu = astropy.io.fits.PrimaryHDU()
    header_hdu.header["NGROUPS"] = seg1[1].data.shape[1]
    header_hdu.header["NINTS"] = seg1[0].header["NINTS"]
    header_hdu.header["NFRAMES"] = 1
    header_hdu.header["GROUPGAP"] = 0
    header_hdu.header["BSTRTIME"] = seg1[0].header["BSTRTIME"]
    header_hdu.header["BENDTIME"] = seg1[0].header["BENDTIME"]
    header_hdu.header["INTSTART"] = seg1[0].header["INTSTART"]
    header_hdu.header["INTEND"] = transit_intstart - 1

    #print(header_hdu.header["INTEND"] - header_hdu.header["INTSTART"], seg1[1].data[:cutoff].shape[0])
    assert(header_hdu.header["INTEND"] - header_hdu.header["INTSTART"] + 1 == seg1[1].data[:cutoff].shape[0])

    output_hdul1 = astropy.io.fits.HDUList([
        header_hdu,
        astropy.io.fits.ImageHDU(seg1[1].data[:cutoff], name="SCI")])
    
    output_hdul1.writeto(output_filename, overwrite=True)
    output_hdul1.close()
    seg1.close()
    return transit_data, transit_intstart, seg1_header

def write_post_transit_segment(input_filename="old_uncalibrated/jw01803001001_04103_00003-seg006_mirimage_uncal.fits", output_filename="jw01803001001_04103_00003-seg005c_mirimage_uncal.fits"):
    seg3 = astropy.io.fits.open(input_filename)
    times = np.linspace(seg3[0].header["BSTRTIME"], seg3[0].header["BENDTIME"], seg3[1].data.shape[0])
    cutoff = transit_end - seg6_begin
    transit_intend = seg3[0].header["INTSTART"] + cutoff
    transit_data2 = seg3[1].data[0:cutoff]

    header_hdu = astropy.io.fits.PrimaryHDU()
    header_hdu.header["NGROUPS"] = seg3[1].data.shape[1]
    header_hdu.header["NINTS"] = seg3[0].header["NINTS"]
    header_hdu.header["NFRAMES"] = 1
    header_hdu.header["GROUPGAP"] = 0
    header_hdu.header["BSTRTIME"] = seg3[0].header["BSTRTIME"]
    header_hdu.header["BENDTIME"] = seg3[0].header["BENDTIME"]
    header_hdu.header["INTSTART"] = transit_intend
    header_hdu.header["INTEND"] = seg3[0].header["INTEND"]
    assert(header_hdu.header["INTEND"] - header_hdu.header["INTSTART"] + 1 == seg3[1].data[cutoff:].shape[0])

    output_hdul3 = astropy.io.fits.HDUList([
        header_hdu,
        astropy.io.fits.ImageHDU(seg3[1].data[cutoff:], name="SCI")])
    
    output_hdul3.writeto(output_filename, overwrite=True)
    output_hdul3.close()
    seg3.close()
    return transit_data2, transit_intend

def write_transit_segment(transit_data, transit_intstart, transit_intend, seg1_header, output_filename="jw01803001001_04103_00003-seg005b_mirimage_uncal.fits"):
    header_hdu = astropy.io.fits.PrimaryHDU()
    header_hdu.header["NGROUPS"] = transit_data.shape[1]
    header_hdu.header["NINTS"] = seg1_header["NINTS"]
    header_hdu.header["NFRAMES"] = 1
    header_hdu.header["GROUPGAP"] = 0
    header_hdu.header["BSTRTIME"] = seg1_header["BSTRTIME"]
    header_hdu.header["BENDTIME"] = seg1_header["BENDTIME"]
    header_hdu.header["INTSTART"] = transit_intstart
    header_hdu.header["INTEND"] = transit_intend - 1

    assert(header_hdu.header["INTEND"] - header_hdu.header["INTSTART"] + 1 == transit_data.shape[0])

    output_hdul2 = astropy.io.fits.HDUList([
        header_hdu,
        astropy.io.fits.ImageHDU(transit_data, name="SCI")])
    output_hdul2.writeto(output_filename, overwrite=True)
    output_hdul2.close()


transit_data, transit_intstart, seg1_header = write_pre_transit_segment()
transit_data2, transit_intend = write_post_transit_segment()

transit_data = np.append(transit_data, transit_data2, axis=0)
write_transit_segment(transit_data, transit_intstart, transit_intend, seg1_header)

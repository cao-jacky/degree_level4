import ppxf_fitter
import voronoi_2d_binning

def cube_runner():
    cf = ppxf_fitter.cat_func()
    catalogue = cf['cat'] # calling sorted catalogue from cataogue function
    bright_objects = cf['bo']

    uc = ppxf_fitter.usable_cubes(catalogue, bright_objects) # usable cubes

    for i_cube in range(len(uc)):
        cube_id = int(uc[i_cube])
        voronoi_2d_binning.voronoi_binning(cube_id)

if __name__ == '__main__':
    cube_runner()

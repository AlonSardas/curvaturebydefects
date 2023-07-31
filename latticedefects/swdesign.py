import numpy as np

from latticedefects.latticegenerator import TriangularLatticeGenerator


def create_lattice_for_sphere_by_traceless_quadrupoles(nx, ny,
                                                       defects_x_jumps=4,
                                                       padding=1,
                                                       factor=0.0001):
    lattice_gen = TriangularLatticeGenerator(nx, ny)

    nx -= 1  # For nx dots we have nx-1 bonds
    middle_x = nx // 2
    max_x = nx - padding + 1
    xs_right = np.arange(middle_x, max_x, defects_x_jumps)
    print(xs_right)
    xs_left = -xs_right[1:] + 2 * middle_x
    xs = np.append(xs_left[::-1], xs_right)
    print(f"defects at {xs}")
    for x in xs:
        y_defects = round((ny - 2 * padding) * factor * (x - nx / 2) ** 2)
        x_index = round(x)
        if y_defects > ny:
            raise RuntimeError(
                "Got more defects than lattice site at column "
                f"{x_index}, ny={ny}, defects={y_defects}"
            )
        print(f"putting in x={x_index}, {y_defects} defects")
        for yy in np.linspace(padding, ny - padding, int(y_defects) + 2)[1:-1]:
            y = round(yy)
            print(f"Putting SW at {x_index},{y}")
            lattice_gen.add_SW_defect(y, x_index)
    return lattice_gen


def create_lattice_for_negative_K_SW(nx, ny,
                                     defects_y_jumps=4,
                                     padding=1,
                                     factor=0.0001):
    lattice_gen = TriangularLatticeGenerator(nx, ny)

    middle_y = ny // 2
    max_y = ny - padding + 1
    ys_right = np.arange(middle_y, max_y, defects_y_jumps)
    ys_left = -ys_right[1:] + 2 * middle_y
    ys = np.append(ys_left[::-1], ys_right) - 1
    print(f"defects at {ys}")
    for y in ys:
        x_defects = round((nx - 2 * padding) * factor * (y - ny / 2) ** 2)
        y_index = round(y)
        if x_defects > nx:
            raise RuntimeError(
                "Got more defects than lattice site at column "
                f"{y_index}, nx={nx}, defects={x_defects}"
            )
        print(f"putting in y={y_index}, {x_defects} defects")
        for xx in np.linspace(padding, nx - padding, int(x_defects) + 2)[1:-1]:
            x = round(xx)
            print(f"Putting SW at {y_index},{x}")
            lattice_gen.add_SW_defect(y_index, x)
    return lattice_gen


def create_cone_by_sw(nx, ny, defects_jumps, padding):
    lattice_gen = TriangularLatticeGenerator(nx, ny)
    for x in range(nx // 2, nx - padding, defects_jumps):
        for y in range(padding, ny - padding, defects_jumps):
            # if (x - nx // 2) >= abs(y - ny // 2):
            if (x - nx // 2) >= abs(y - ny // 2):
                print(f"Putting SW at {x},{y}")
                try:
                    lattice_gen.add_SW_defect(y, x)
                except RuntimeError as e:
                    print("warning: ", e.args)

    return lattice_gen

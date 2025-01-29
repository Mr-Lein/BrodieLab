################################################################################################################
# Utility routines
# Author: R. Bourquard
################################################################################################################

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np



def plot_points(title, x_axis_label, y_axis_label, point_label, x_points, y_points):
    """
    ##################################################################################

    Plots a set of (x,y) points

    Args:
        title:         The title for the plot
        x_axis_label:  The x-axis label
        y_axis_label:  The y-axis label
        point_label:   The label for the points.  Specifying null ("") causes no points
                         to be plotted.
                         The first 3 characters specify the point style:
                         The 1st character specifies the color ('g'=green, 'r'=red, 'b'=blue etc)
                         The 2nd character specifies the marker style ('.'=point, 'o'=circle, '+'=plus, etc)
                         The 3rd character is not used and is typically set to '|' for reading clarity
                         For details, see the formatting used by matplotlib.pyplot.plot
        x_points:      The array of x values for the points.
        y_points:      The array of matching y values for the points.
    Returns:
        none
    ##################################################################################
    """
    nPoints = len(x_points)
    print(title,' # of data points =', nPoints)
    print('CLOSE PLOT TO CONTINUE')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    style = point_label[:3]
    if style[2] == "|":
        # remove the unneeded closing "|"
        style = style[:2]
    label = point_label[3:]
    plt.plot(x_points, y_points, style, label=label)
    plt.ylabel(y_axis_label)
    plt.xlabel(x_axis_label)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()




def plot_points_and_a_line(title, x_axis_label, y_axis_label, point_label, x_points, y_points,
                           line_label, x_line, y_line) -> object:
    """
    ##################################################################################

    Plots the input points (given by x,y values) and a line (given by x,y values)

    Args:
        title:        The title for the plot.
        x_axis_label: The x-axis label.
        y_axis_label: The y-axis label.
        point_label:  The label for the points.  Specifying null ("") causes no points
                        to be plotted.
                        The first 3 characters specify the point style:
                        The 1st character specifies the color ('g'=green, 'r'=red, 'b'=blue etc)
                        The 2nd character specifies the marker style ('.'=point, 'o'=circle, '+'=plus, etc)
                        The 3rd character is not used and is typically set to '|' for reading clarity
                        For details, see the formatting used by matplotlib.pyplot.plot
        x_points:     The array of x values for the points.
        y_points:     The array of matching y values for the points.
        line_label:   The label for the line.  Specifying null ("") causes no line
                        to be plotted.
                        The first 3 characters specify the line style:
                        The 1st character specifies the color ('g'=green, 'r'=red, 'b'=blue etc)
                        The 2nd character specifies the marker style ('.'=point, 'o'=circle, '+'=plus, etc)
                        The 3rd character specifies the line style ('-'=solid, or ':'=dotted)
                        If markers are not desired, shift the line style to the 2nd character position, and set the
                        3rd character = "|".  For example: a black dotted line = 'k:|', a red dashed line = 'r--'.
                        For details, see the formatting used by matplotlib.pyplot.plot
        x_line:       The array of x values for the line.  The points must be given in
                        the order in which they are to be connected.
        y_line:       The array of matching y values for the line.
    Returns:
        none
    ##################################################################################
    """
    print('CLOSE PLOT TO CONTINUE')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    # plot the lines
    if line_label != "":
        # There is a line to plot
        style = line_label[:3]
        if style[2]=="|":
            # remove the unneeded closing "|"
            style = style[:2]
        label = line_label[3:]
        plt.plot(x_line, y_line, style, label=label)
    # plot the points
    if point_label != '':
        style = point_label[:3]
        if style[2] == "|":
            # remove the unneeded closing "|"
            style = style[:2]
        label = point_label[3:]
        plt.plot(x_points, y_points, style, label=label)
    plt.ylabel(y_axis_label)
    plt.xlabel(x_axis_label)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()




def plot_points_and_lines(title: object, x_axis_label: object, y_axis_label: object, point_label: object, x_points: object, y_points: object,
                          line_labels: object, x_lines: object, y_lines: object) -> object:
    """
    ##################################################################################

    Plots the input points (given by x,y values) and a line (given by x,y values)

    Args:
        title:        The title for the plot.
        x_axis_label: The x-axis label.
        y_axis_label: The y-axis label.
        point_label:  The label for the points.  Specifying null ("") causes no points
                        to be plotted.
                        The first 3 characters specify the point style:
                        The 1st character specifies the color ('g'=green, 'r'=red, 'b'=blue etc)
                        The 2nd character specifies the marker style ('.'=point, 'o'=circle, '+'=plus, etc)
                        The 3rd character is not used but is typically set to '|' for reading clarity
                        For details, see the formatting used by matplotlib.pyplot.plot
        x_points:     The array of x values for the points.
        y_points:     The array of matching y values for the points.
        line_labels:  An array of labels for the lines.  Specifying null ("") causes no
                        line to be plotted.
                        The first 3 characters specify the line style:
                        The 1st character specifies the color ('g'=green, 'r'=red, 'b'=blue etc)
                        The 2nd character specifies the marker style ('.'=point, 'o'=circle, '+'=plus, etc)
                        The 3rd character specifies the line style ('-'=solid, or ':'=dotted)
                        If markers are not desired, shift the line style to the 2nd character position, and set the
                        3rd character = "|".  For example: a black dotted line = 'k:|', a red dashed line = 'r--'.
                        For details, see the formatting used by matplotlib.pyplot.plot
        x_lines:      A 2-D array of x values for the line.  Each row is a new line,
                        each column is an x value.  The points must be given
                        in the order in which they are to be connected.
        y_lines:      A 2-D array of matching y values for the lines.
    Returns:
        none
    ##################################################################################
    """
    print('CLOSE PLOT TO CONTINUE')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')

    # plot the lines
    if line_labels != "":
        # there are lines to plot
        nLines = len(x_lines)
        # Plot each line
        for iLine in range(nLines):
            current_full_label = line_labels[iLine][:]
            current_style = current_full_label[:3]
            if current_style[2]=="|":
                # remove the unneeded closing "|"
                current_style = current_style[:2]
            current_label = current_full_label[3:]
            plt.plot(x_lines[iLine][:], y_lines[iLine][:], current_style, label=current_label)

    # plot the points
    if point_label != '':
        style = point_label[:3]
        if style[2] == "|":
            # remove the unneeded closing "|"
            style = style[:2]
        label = point_label[3:]
        plt.plot(x_points, y_points, style, label=label)
    plt.ylabel(y_axis_label)
    plt.xlabel(x_axis_label)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()




def plot_points_and_slopes(title, x_axis_label, y_axis_label, point_label,
                           x_points, y_points, line_labels, slopes, bs):
    """
    ##################################################################################

    Plots the input points (given by x,y) and some lines (given by slope, intercept)

    Args:
        title:        The title for the plot.
        x_axis_label: The x-axis label.
        y_axis_label: The y-axis label.
        point_label:  The label for the points.  Specifying null ("") causes no points
                        to be plotted.
                        The first 3 characters specify the point style:
                        The 1st character specifies the color ('g'=green, 'r'=red, 'b'=blue etc)
                        The 2nd character specifies the marker style ('.'=point, 'o'=circle, '+'=plus, etc)
                        The 3rd character is not used but is typically set to '|' for reading clarity
                        For details, see the formatting used by matplotlib.pyplot.plot
        x_points:     The array of x values for the points.
        y_points:     The array of matching y values for the points.
        line_labels:  An array of labels for the lines.  Specifying null ("") causes no
                        line to be plotted.
                        The first 3 characters specify the line style:
                        The 1st character specifies the color ('g'=green, 'r'=red, 'b'=blue etc)
                        The 2nd character specifies the marker style ('.'=point, 'o'=circle, '+'=plus, etc)
                        The 3rd character specifies the line style ('-'=solid, or ':'=dotted)
                        If markers are not desired, shift the line style to the 2nd character position, and set the
                        3rd character = "|".  For example: a black dotted line = 'k:|', a red dashed line = 'r--'.
                        For details, see the formatting used by matplotlib.pyplot.plot
        slopes:       An array of the slopes for the lines.
        bs:           An array of the b values for the lines.
    Returns:
        none
    ##################################################################################
    """
    nPoints = len(x_points)
    nLines = len(slopes)
    print('CLOSE PLOT TO CONTINUE')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    style = point_label[:3]
    if style[2] == "|": style = style[:2]
    label = point_label[3:]
    plt.plot(x_points, y_points, style, label=label)
    for iLine in range(nLines):
        line_y_points = []
        for iPoint in range(nPoints):
            #Plot the line over the x-extent of the points
            line_y_points.append(slopes[iLine]*x_points[iPoint] + bs[iLine])
        if line_labels[0] == '':
            plt.plot(x_points, line_y_points, ':')
        else:
            plt.plot(x_points, line_y_points, ':', label=line_labels[iLine])
    plt.ylabel(y_axis_label)
    plt.xlabel(x_axis_label)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()




def plot_points_on_a_surface(title, x_label, y_label, point_label, x_points, y_points, x_grid, y_grid, z_grid):
    """
    ##################################################################################

    Plots points on a flat contour map as red 'x's

    Args:
        title:        The title for the plot.
        x_axis_label: The x-axis label.
        y_axis_label: The y-axis label.
        point_label:  The label for the points.  Specifying null ("") causes no points
                        to be plotted.  Specifying anything else causes the points to
                        be plotted, but point_label's text is ignored.
        x_points:     The array of x values for the points.
        y_points:     The array of matching y values for the points.
        x_grid:       The 2-D grid of x values for the surface.  This grid is often created by np.meshgrid.
        y_grid:       The 2-D grid of y values for the surface.  This grid is often created by np.meshgrid.
        z_grid:       The 2-D grid of z values for the surface, which will be represented by colors.
    Returns:
        none
    """
    print('CLOSE PLOT TO CONTINUE')
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(x_grid, y_grid, z_grid, locator=ticker.LogLocator())
    fig.colorbar(cp)  # Add a colorbar to the plot
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    if point_label != "":
        plt.plot(x_points, y_points, 'rx')
    plt.show()




def plot_a_3D_surface(title, x_label, y_label, z_label, x_grid, y_grid, z_grid):
    """
    ##################################################################################

    Plots a surface within a 3-D cube

    Args:
        title:        The title for the plot.
        x_label:      The x-axis label.
        y_label:      The y-axis label.
        z_label:      The z-axis label.
        x_grid:       The 2-D grid of x values for the surface.  This grid is often created by np.meshgrid.
        y_grid:       The 2-D grid of y values for the surface.  This grid is often created by np.meshgrid.
        z_grid:       The 2-D grid of z values for the surface.
    Returns:
        none
    ##################################################################################
    """
    print('CLOSE PLOT TO CONTINUE')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.show()




def plot_decision_boundary(model, X, y):
    """
    ##################################################################################

    Plots a decision boundary

    Args:
        model:  A function which defines the model value for any (x,y) location.
        X:      2-D array of (x,y) data points
        y:      Array of the color group that each (x,y) data point belongs to, such
                  as 0=inside and 1=outside.
    Returns:
         none
    ##################################################################################
    """
    print('CLOSE PLOT TO CONTINUE')
    # Set min and max values for the plot and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1

    # Generate a grid of points with distance h between them
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


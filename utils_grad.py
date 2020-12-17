import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def plot_grad_wrt_x_or_z(model, sess, x, y, eps, savepath, title): 
    for plotgrad in ['x', 'z']:
        figpath = savepath + 'wrt_{}.png'.format(plotgrad)
        figtitle = title + ',wrt={}'.format(plotgrad)
        if plotgrad == 'x': 
            c_grad = sess.run(model.X_grad, feed_dict={model.x_eval:x, model.y_eval:y})
            c_input = x
        elif plotgrad == 'z': 
            c_grad, c_input = sess.run([model.z_grad, model.z_eval], feed_dict={model.x_eval:x, model.y_eval:y})

        sign_grad = np.sign(c_grad)
        p_rand = get_random_perpendicular_vector(sign_grad)
        grid_x, grid_y, grid_z, grid_l = get_loss_grid(model, sess, c_input, y,
            grad=c_grad, rand=p_rand, 
            eps=eps, num_points=17, plotgrad=plotgrad)
        
        if np.prod(np.shape(x)) == 784: 
            plot_grid(figpath, np.reshape(x, [28,28]), grid_x, grid_y, grid_z, grid_l, figtitle)
        else:
            plot_grid(figpath, np.reshape(x, [32,32,3]), grid_x, grid_y, grid_z, grid_l, figtitle)


def get_random_perpendicular_vector(grad, random_state=6789):
    org_shape = np.shape(grad)
    grad = np.reshape(grad, [1,-1])
    random = np.random.RandomState(random_state)
    rand = random.choice([1, -1], grad.shape)
    assert(len(grad.shape)==2)
    assert(np.shape(grad)[0]==1)

    while np.sum(rand * grad) != 0:
        if np.sum(rand * grad) % 2 == 1:
            idx = random.randint(rand.shape[1])
            rand[0, idx] = 0
        else:
            # modify an entry to increase or decrease the dot product
            idx = random.randint(rand.shape[1])
            if np.sum(rand * grad) < 0 and rand[0, idx] * grad[0, idx] < 0:
                rand[0, idx] = -rand[0, idx]
            elif np.sum(rand * grad) > 0 and rand[0, idx] * grad[0, idx] > 0:
                rand[0, idx] = -rand[0, idx]
    return np.reshape(rand, org_shape)

def get_loss_grid(model, sess, image, label, grad, rand, eps=0.5, num_points=17, plotgrad='x'): 
    if type(eps) is not list:
        x = np.outer(np.linspace(-eps, eps, num_points), np.ones(num_points))
    else: 
        x = np.outer(np.linspace(eps[0], eps[1], num_points), np.ones(num_points))
    y = x.copy().T

    x_flatten = x.flatten()
    y_flatten = y.flatten()
    N = len(x_flatten)

    # assert(np.max(image) < 2) 
    rand = rand #/ 255. 
    grad = grad #/ 255. 

    neighbor_labels = [label] * N
    neighbor_images = np.zeros((N, *image.shape[1:]))
    for t, (i, j) in enumerate(zip(x_flatten, y_flatten)):
        neighbor_images[t] = image + i * rand + j * grad

    neighbor_images = np.expand_dims(neighbor_images, axis=1) # [N, 1, d]

    z = np.zeros(N)
    cl = np.zeros(N)
    for i in range(N):
        if plotgrad == 'x': 
            z[i], cl[i] = sess.run([model.mean_xent_eval, model.X_cls],feed_dict={model.x_eval: neighbor_images[i], model.y_eval: neighbor_labels[i]})
        elif plotgrad == 'z': 
            z[i], cl[i] = sess.run([model.mean_xent_eval, model.X_cls],feed_dict={model.z_eval: neighbor_images[i], model.y_eval: neighbor_labels[i]})

    z = z.reshape(num_points, num_points)
    cl = cl.reshape(num_points, num_points)

    return x, y, z, cl    

def normalize_image(image): 
    print('normalize_image, image range: ', np.max(image), np.min(image))
    if image.dtype in ['float32', 'float64']: 
        assert(np.max(image) >= 100.) 
        image = image / 255. 

    elif image.dtype in ['int32', 'int64']: 
        assert(np.max(image) == 255)
        image = image.astype(np.float32)
        image = image / 255.

    return image

def plot_grid(figname, image, x, y, z, cl, title): 
    max_z = np.max(z)
    cmap = ListedColormap(['blue', 'orange', 'green', 'red', 'purple', 
        'brown', 'pink', 'gray', 'olive', 'cyan'])

    # visualize
    size = 5
    fig = plt.figure(figsize=(size * 3, size))

    # first subplot
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(normalize_image(image))
    ax.grid(False)
    ax.axis('off')
    ax.set_title('Input image')

    # second subplot
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.plot_surface(y, x, z, cmap=cm.coolwarm, linewidth=0, antialiased=False, shade=True, alpha=0.5)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_zlim(0, max_z)
    ax.set_xlabel('Grad')
    ax.set_ylabel('Rand')
    ax.set_title('Prediction Surface')

    # Third subplot 
    ax = fig.add_subplot(1, 3, 3)
    psm = ax.pcolormesh(cl, cmap=cmap, rasterized=True, vmin=0, vmax=10)
    fig.colorbar(psm, ax=ax)
    start, end = ax.get_xlim()
    ax.set_xticks(np.arange(start, end, (start-end)/10.))
    ax.set_xticklabels([str(t) for t in np.arange(start, end, (start-end)/10.)])
    start, end = ax.get_ylim()
    ax.set_yticks(np.arange(start, end, (start-end)/10.))
    ax.set_yticklabels([str(t) for t in np.arange(start, end, (start-end)/10.)])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('Grad')
    ax.set_ylabel('Rand')
    ax.set_title(title)  

    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()

def plot_mul_grid(figname, image, xes, yes, zes, cles, titles): 
    nb = 1 + len(zes)

    size = 5 
    fig = plt.figure(figsize=(size*nb, 2*size)) # 2 row and nb col 

    ax = fig.add_subplot(2, nb, 1)
    ax.imshow(normalize_image(image))
    ax.grid(False)
    ax.axis('off')
    ax.set_title('Input image')
    cmap = ListedColormap(['blue', 'orange', 'green', 'red', 'purple', 
        'brown', 'pink', 'gray', 'olive', 'cyan'])
    for i , (x, y, z, cl, title) in enumerate(zip(xes, yes, zes, cles, titles)): 
        max_z = np.max(z)
        ax = fig.add_subplot(2, nb, i+2, projection='3d')
        ax.plot_surface(y, x, z, cmap=cm.coolwarm, linewidth=0, antialiased=False, shade=True, alpha=0.5)

        ax.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_zlim(0, max_z)
        ax.set_xlabel('Grad')
        ax.set_ylabel('Rand')
        ax.set_title(title)

        ax = fig.add_subplot(2, nb, i+2+nb)
        psm = ax.pcolormesh(cl, cmap=cmap, rasterized=True, vmin=0, vmax=10)
        fig.colorbar(psm, ax=ax)
        start, end = ax.get_xlim()
        ax.set_xticks(np.arange(start, end, (start-end)/10.))
        ax.set_xticklabels([str(t) for t in np.arange(start, end, (start-end)/10.)])
        start, end = ax.get_ylim()
        ax.set_yticks(np.arange(start, end, (start-end)/10.))
        ax.set_yticklabels([str(t) for t in np.arange(start, end, (start-end)/10.)])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_xlabel('Grad')
        ax.set_ylabel('Rand')
        ax.set_title(title)        

    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()            

def plot_mul_grid_with_adv(figname, image, adv, xes, yes, zes, cles, titles): 
    nb = 1 + len(zes)

    size = 5 
    fig = plt.figure(figsize=(size*nb, 2*size)) 

    ax = fig.add_subplot(2, nb, 1)
    ax.imshow(normalize_image(image))
    ax.grid(False)
    ax.axis('off')
    ax.set_title('Input image')

    ax = fig.add_subplot(2, nb, 1+nb)
    ax.imshow(normalize_image(adv))
    ax.grid(False)
    ax.axis('off')
    ax.set_title('Adversarial image')

    cmap = ListedColormap(['blue', 'orange', 'green', 'red', 'purple', 
        'brown', 'pink', 'gray', 'olive', 'cyan'])

    for i , (x, y, z, cl, title) in enumerate(zip(xes, yes, zes, cles, titles)): 
        max_z = np.max(z)
        ax = fig.add_subplot(2, nb, i+2, projection='3d')
        ax.plot_surface(y, x, z, cmap=cm.coolwarm, linewidth=0, antialiased=False, shade=True, alpha=0.5)

        ax.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_zlim(0, max_z)
        ax.set_xlabel('Grad')
        ax.set_ylabel('Rand')
        ax.set_title(title)

        ax = fig.add_subplot(2, nb, i+2+nb)
        psm = ax.pcolormesh(cl, cmap=cmap, rasterized=True, vmin=0, vmax=10)
        fig.colorbar(psm, ax=ax)
        start, end = ax.get_xlim()
        ax.set_xticks(np.arange(start, end, (start-end)/10.))
        ax.set_xticklabels([str(t) for t in np.arange(start, end, (start-end)/10.)])
        start, end = ax.get_ylim()
        ax.set_yticks(np.arange(start, end, (start-end)/10.))
        ax.set_yticklabels([str(t) for t in np.arange(start, end, (start-end)/10.)])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_xlabel('Grad')
        ax.set_ylabel('Rand')
        ax.set_title(title) 
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close() 

def save_image(image, savepath): 
    plt.figure()
    plt.imshow(normalize_image(image))
    plt.axis('off')    
    plt.savefig(savepath, dpi=300)
    plt.close()

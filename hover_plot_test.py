import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)


# data to be plotted
x = np.random.rand(15)
y = np.random.rand(15)
names = np.array(list("ABCDEFGHIJKLMNO"))
c = np.random.randint(1,5,size=15)

norm = plt.Normalize(1,4)
cmap = plt.cm.RdYlGn

fig,ax = plt.subplots()
sc = ax.scatter(x,y,c=c, s=100, cmap=cmap, norm=norm)



# predefined annotations
annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)


def plot_update_callback(ind):
    
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                           " ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)
    

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax: # checks if the mouse/"event" is inside the desired plot axis
        cont, ind = sc.contains(event) 
        if cont: # this checks if the mouse/"event" is over any of the data in the scatter by 'cont'
            plot_update_callback(ind) # updates the plot annotations by the index found
            annot.set_visible(True) # set the annotations visible
            fig.canvas.draw_idle() # redraws the plot when able to
        else: # else happens when mouse is not over any point in the plot
            if vis: # if anything was visible turn it off ie. if the mouse went off of a data point
                annot.set_visible(False) # hides the annotations
                fig.canvas.draw_idle() # redraws the plot when able to 



fig.canvas.mpl_connect("motion_notify_event", hover) # checks the open plot window for mouse events and triggers the "hover" callback function on event

plt.show()
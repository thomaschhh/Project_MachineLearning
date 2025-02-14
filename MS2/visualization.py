import numpy as np
import matplotlib.pyplot as plt
import os


# train model
# hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_split=0.2)

def plot_loss_acc(train_loss, val_loss, train_acc, val_acc, now, n_epochs, k, tsamples, eps, outfile = True):
    # visualizing losses and accuracy
    #train_loss, val_loss = hist.history['loss'], hist.history['val_loss']
    #train_acc, val_acc = hist.history['accuracy'], hist.history['val_accuracy']

    # setup plot
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15,5))

    # plot loss
    ax[0].plot(range(n_epochs), train_loss)
    ax[0].plot(range(n_epochs), val_loss)
    ax[0].set_ylabel('loss')
    ax[0].set_title('train_loss vs val_loss')
    ax[0].set_title('train_loss')

    # plot accuracy
    ax[1].plot(range(n_epochs), train_acc)
    ax[1].plot(range(n_epochs), val_acc)
    ax[1].set_ylabel('accuracy')
    ax[1].set_title('train_acc vs val_acc')
    ax[1].set_title('train_acc')

    # plot adjustement
    for a in ax:
        a.grid(True)
        #a.legend(['train','val'],loc=4)
        a.legend(['train'],loc=4)
        a.set_xlabel('Number of Epochs')
    if outfile:
        if not os.path.exists(f"output/figures/training_k{k}_s{tsamples}_e{n_epochs}"):
            os.mkdir(f"output/figures/training_k{k}_s{tsamples}_e{n_epochs}")
        plt.savefig(f"output/figures/training_k{k}_s{tsamples}_e{n_epochs}/loss_plot_{n_epochs}.png")
    plt.show()
    plt.close('all') 
    
    
def show_img(batch, name = "Input_3x3.png", outfile = True):
    # load data
      
    num_im = 9
 
    #### Tensor Images ####
    # Normalizing and clipping for imshow

    
    plt.figure()
    for i, item in enumerate(batch):
        inp = item.permute(1, 2, 0).clone().detach().cpu().numpy()#.reshape(112, 112, 3)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        
        # Plotting
        ax = plt.subplot(3,3,i+1)
        im = ax.imshow(inp)
        plt.title(f"{i}-th input picture")
        plt.axis('off')
        
       
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle('9 Randomly Chosen Input Pictures')
    plt.savefig(name)
    plt.show()

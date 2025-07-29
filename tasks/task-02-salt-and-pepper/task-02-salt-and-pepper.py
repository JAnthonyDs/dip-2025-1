import numpy as np

def create_salt_and_pepper_noise(height=100, width=100, salt_prob=0.05, pepper_prob=0.05):
    """
    Returns an image ∈ [-1, 1] containing salt (I = 1.0) and 
    pepper (I = -1.0) noise with respective probability distributions
    equal to salt_prob and pepper_prob. Pixels without noise have values of 0.5.
    """
    ### START CODE HERE ###

    #Initializing a base image with value 0.5
    img = np.ones((height, width)) * 0.5 
    
    salt_mask = np.random.random((height, width)) < salt_prob # Matriz de ruído salt
    img[salt_mask] = 1.0 # Aplica salt na imagem
    
    remaining_pixels = ~salt_mask # Aqui temos apenas os píxels onde não foi aplicado o Salt
    pepper_prob_adjusted = pepper_prob / (1 - salt_prob) # Ajuste de probabilidade 
    pepper_mask = (np.random.random((height, width)) < pepper_prob_adjusted) & remaining_pixels # Aplicação do Ruído pepper
    img[pepper_mask] = -1.0

    ### END CODE HERE ###
    return img

def main():
    
    img = create_salt_and_pepper_noise(100, 100, 0.1, 0.1)
    
    salt_count = np.sum(img == 1.0)
    pepper_count = np.sum(img == -1.0)
    
    print(f"Salt pixels: {salt_count}, Pepper pixels: {pepper_count}")
    
    assert 900 <= salt_count <= 1100, "Salt pixel count is outside expected range."
    assert 900 <= pepper_count <= 1100, "Pepper pixel count is outside expected range."
    
    print("Test passed!")


if __name__ == "__main__":

    main()
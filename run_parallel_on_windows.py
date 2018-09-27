"""
 IMPORTANT: Copy into taskX folder, otherwise import path for read and write incorrect
"""

# import your file
import task0.task0_franz as tf

# AMD GPU support (not sure if working)
# import plaidml.keras
# plaidml.keras.install_backend()

# run in main cause windows is dumb...
if __name__ == '__main__':
    tf.go()

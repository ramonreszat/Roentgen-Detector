from mxnet import nd

def box_iou(A,G):

  # conversion to min/max rectangle span
  #     ymin = y - h/2
  #     xmin = x - w/2
  #     ymax = y + h/2
  #     xmax = x + w/2
  Gymin = G[:,:,1,:,:] - G[:,:,3,:,:]/2
  Gxmin = G[:,:,0,:,:] - G[:,:,2,:,:]/2
  Gymax = G[:,:,1,:,:] + G[:,:,3,:,:]/2
  Gxmax = G[:,:,0,:,:] + G[:,:,2,:,:]/2

  Aymin = A[:,:,1,:,:] - A[:,:,3,:,:]/2
  Axmin = A[:,:,0,:,:] - A[:,:,2,:,:]/2
  Aymax = A[:,:,1,:,:] + A[:,:,3,:,:]/2
  Axmax = A[:,:,0,:,:] + A[:,:,2,:,:]/2

  # Ai
  dx = nd.minimum(Gxmax,Axmax) - nd.maximum(Gxmin,Axmin)
  dy = nd.minimum(Gymax,Aymax) - nd.maximum(Gymin,Aymin)

  #if dx or dy is negative no intersection else dx hadamard dy
  Ai = nd.multiply(nd.relu(dx),nd.relu(dy))
  

  # Au = s^2 + wh - Ai > 0
  Au = nd.multiply(A[:,:,2,:,:],A[:,:,3,:,:]) + nd.multiply(G[:,:,2,:,:],G[:,:,3,:,:]) - Ai

  return nd.relu(nd.divide(Ai,Au))
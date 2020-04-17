def find_threshold(loader):
    
    model_SNN = copy.deepcopy(model_ANN)   #copy model weights
    V = np.zeros(len(model.module.features) + len(model.module.classifier))
    
    for l in range(1,np.shape(V)):
        v = 0
        for t in range(timesteps):
            out = []
            for batch_idx, (data, target) in enumerate(loader):
                out.append(torch.mul(torch.le(torch.rand_like(data), torch.abs(data)*1.0).float(),torch.sign(data)))
                
                if torch.cuda.is_available() and use_cuda:
                    data, target = data.cuda(), target.cuda()

                with torch.no_grad():
                    model.eval()
                    model.module.network_init(timesteps)
                    output = model(data, 0, find_max_mem=True, max_mem_layer=layer)

                    if output>max_act:
                        max_act = output.item()
                    f.write('\nBatch:{} Current:{:.4f} Max:{:.4f}'.format(batch_idx+1,output.item(),max_act))
                    if batch_idx==0:
                        ann_thresholds[pos] = max_act
                        pos = pos+1

                        model.module.threshold_init(scaling_threshold=scaling_threshold, reset_threshold=reset_threshold, thresholds = ann_thresholds[:], default_threshold=default_threshold)
                        break
                
                for k in range(l):
                    if k<l:
                        #execute Algorithm 3
                    else:
                        output = model(out[])  # do complete this
                        if np.max(A)> v:
                            v= np.max(A)
                    
                
        V[l] = v
    
    return V
                        

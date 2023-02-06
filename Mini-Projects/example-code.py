 def _detectApex(self,IM_ROI2_grey,line_image_peak,arrow_x1,arrow_y1,h,v):
        # Isolate the apex
        offset = Arrow_Detector.ENV['DETECTION_APEX_OFFSET']
        IM_ROI_APEX = IM_ROI2_grey[arrow_y1-offset:arrow_y1+offset,arrow_x1-offset:arrow_x1+offset]
        IM_ROI_LINE = line_image_peak[arrow_y1-offset:arrow_y1+offset,arrow_x1-offset:arrow_x1+offset] 
        IM_ROI_APEX_edges = cv2.Canny(IM_ROI_APEX,50,100)
        IM_ROI_APEX_masekd = cv2.multiply(IM_ROI_LINE,IM_ROI_APEX_edges)
        
        #GUI.imShow(IM_ROI_APEX)
        #GUI.imShow(IM_ROI_APEX_edges)

        contours_line, hierarchy_line = cv2.findContours(IM_ROI_APEX_masekd.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours_line) == 0:
            return None, None, None, None,None,None,None,None,None,None
        
        max_contour_idx = getMaxContourIdx(contours_line)
        xxx,yyy,www,hhh = cv2.boundingRect(contours_line[max_contour_idx])

        #GUI.imShow(Image_Tools.debugRectangle(IM_ROI_APEX_masekd,xxx,yyy,www,hhh))
        #GUI.imShow(IM_ROI_APEX_masekd)

        IM_ROI_APEX_clipped = np.zeros(IM_ROI_APEX_masekd.shape, "uint8")
        IM_ROI_APEX_clipped[yyy:yyy+hhh,xxx:xxx+www] = IM_ROI_APEX_masekd[yyy:yyy+hhh,xxx:xxx+www] 

        IM_ROI_APEX_masekd = IM_ROI_APEX_clipped
        #GUI.imShow(IM_ROI_APEX_clipped)

        # respect orientation
        y,x = np.where(IM_ROI_APEX_masekd > 1)
        np.sort(y)
        #print(h)
        #print(v)
        if h == 'l':
            if v == 'u':
                arrow_y2 = y[y.shape[0]-1]
                x = np.where(IM_ROI_APEX_masekd[arrow_y2,:] > 1)[0]
                np.sort(x)
                arrow_x2 = x[x.shape[0]-1]
            else:
                arrow_y2 = y[0]
                x = np.where(IM_ROI_APEX_masekd[arrow_y2,:] > 1)[0]
                arrow_x2 = x[x.shape[0]-1]
        else:
            if v == 'u':
                arrow_y2 = y[y.shape[0]-1]
                x = np.where(IM_ROI_APEX_masekd[arrow_y2,:] > 1)[0]
                np.sort(x)
                arrow_x2 = x[0]
                #arrow_y2 = yyy
                #arrow_x2 = xxx
            else:
                arrow_y2 = y[0]
                x = np.where(IM_ROI_APEX_masekd[arrow_y2,:] > 1)[0]
                np.sort(x)
                arrow_x2 = x[0]   
        
        # transform to original space
        arrow_y1 = (arrow_y1 - offset) + arrow_y2
        arrow_x1 = (arrow_x1 - offset) + arrow_x2

        return arrow_x1,arrow_y1,IM_ROI_APEX
clear; close all

%% for tectonics
load ./pickleFiles/all_data_wo_seasn;
stns = stns';

fileID = fopen('helper_files/twn_inland.txt','r');
formatSpec = '%f %f'; %defining the format of the data
sizeA = [2 Inf]; %defining the size of the data
A = fscanf(fileID,formatSpec,sizeA); %reading the data using fscanf function
fclose(fileID); %closing the file

data={dU, dN, dE};
dtnm={'U', 'N', 'E'};
for mode=1
%     close all;
    for dd=1:length(data)
        %For dN
        [vp, var_porc, eof_p_n, exp_coef_n] = eof_n_optimizado_A2(data{dd});

        pcdata=exp_coef_n(:,mode);
        spatialdata=eof_p_n(:,mode);
        save(sprintf('MATLAB_output/with_tr_jmp_pc%d_%s',mode, dtnm{dd}),'tdata','pcdata');
        Adata=[tdata;pcdata'];

        fileID = fopen(sprintf('MATLAB_output/with_tr_jmp_pc%d_%s.txt',mode, dtnm{dd}),'w');
        fprintf(fileID,'%6s %12s\n','tdata','pcdata');
        fprintf(fileID,'%6.2f %12.8f\n',Adata);
        fclose(fileID);
        

        % % %Visualization of the spatial pattern
        xq=min(slon):0.01:122.5;%max(slon);
        yq=19.8:0.01:25.6;
        [X,Y]=meshgrid(xq,yq);
        F = scatteredInterpolant(slon',slat',eof_p_n(:,mode),'natural','linear');
        vq= F(X,Y);
        [r,c]=size(vq);

        [in,on] = inpolygon(X,Y,A(1,:),A(2,:)); 
        inon = in | on;                                           
        [rin,cin]=size(inon);

        for i=1:rin
            for j=1:cin
                if (inon(i,j)==0 && inon(i,j)==0)
                    X(i,j)=NaN;
                    Y(i,j)=NaN;
                    vq(i,j)=NaN;
                end
            end 
        end

        T = table(stns,slon',slat',spatialdata);
        writetable(T,sprintf('MATLAB_output/with_tr_jmp_spatial%d_%s.txt',mode, dtnm{dd}),'Delimiter',' ')
        

        %% Using m_map
        figure()
        m_proj('miller','lat',[21.8,25.5],'lon',[119.5,122.5]);
        
        
%         m_pcolor(X,Y,(vq)); shading interp; colormap('jet');

        colorMap1 = [ones(256/4,1),linspace(0,1,256/4)', zeros(256/4,1)];
        colorMap2 = [ones(256/4,1),ones(256/4,1),linspace(0,1,256/4)'];
        colorMap3 = [linspace(1,0,256/4)',ones(256/4,1),linspace(1,0,256/4)'];
        colorMap4 = [zeros(256/4,1),linspace(1,0,256/4)',linspace(0,1,256/4)'];
        colorMap=[colorMap1; colorMap2; colorMap3; colorMap4];
%         m_pcolor(X,Y,(vq)); shading interp; colormap(flipud(colorMap));
        m_pcolor(X,Y,(vq)); shading interp; colormap(colorMap);


        selected_stations = {'PEPU' 'DAJN' 'NDHU' 'CHUN' 'SHUL' 'TUNH' 'DAWU' 'CHGO' 'YENL' 'SHAN' 'SOFN' 'TAPE' 'ERPN' 'CHEN' 'TAPO' 'SINL' 'LONT' 'JULI' 'JSUI' 'TTUN' 'NAAO' 'SPAO' 'MOTN' 'SLNP' 'WARO' 'SLIN' 'WULU'};

        
        for i=1:length(slon)
            idx = find(ismember(selected_stations, stns(i,:)));
            if length(idx) >= 1
                m_line(slon(i),slat(i),'marker','v','color',[0 0 0.5],'linest','none','markerfacecolor','b','markersize',10,'linewi',0.5);
            else
                m_line(slon(i),slat(i),'marker','v','color',[0 0 0.5],'linest','none','markerfacecolor','w','markersize',10,'linewi',0.5);

            end
        end
        m_line(A(1,:),A(2,:),'color','k','linewi',1);
        
        m_grid('box','fancy','linestyle','none','fontsize',16,'FontName','Times New Roman','backcolor',[1 1 1]);

        h=colorbar('eastoutside');
        maxmin=[min(spatialdata) max(spatialdata)];
        caxis([-max(abs(maxmin)) max(abs(maxmin))])
%         caxis([-1 1])
    %     h.Label.String = 'EOF ';
        % set(h,'FontSize',8)
        title(sprintf('%s, Variance: %.1f %%',dtnm{dd},var_porc(mode)),'fontsize',22)

        ax = gca;
        outerpos = ax.OuterPosition;
        ti = ax.TightInset;
        left = outerpos(1) + ti(1);
        bottom = outerpos(2) + ti(2);
        ax_width = outerpos(3) - ti(2) - ti(3);
        ax_height = outerpos(4) - ti(2) - ti(4);
        ax.Position = [left bottom ax_width ax_height];
        print('-dpng','-r600',[sprintf('MATLAB_output/with_tr_jmp_eof%d_CGPS_comp_spatial',mode), dtnm{dd}]);
        
        figure('rend','painters','pos',[100 100 900 200])
        plot(tdata,pcdata,'k','linewidth',1)
        xlabel('t','FontSize',22)
        ylabel('Amplitude (in mm)','FontSize',22)
        grid on;
        print('-dpng','-r600',[sprintf('MATLAB_output/with_tr_jmp_eof%d_comp_temp_amp',mode), dtnm{dd}]);
  

    end
end

plot(x3,O,x3,Y3, 'linewidth',2.5);
set(gca,'fontsize',25);
str = '||.||_2 = ';
err = num2str(err);
str = strcat('Approximation, ',str,err);
legend(str,'Actual');

str1 = 'N = ';

str1 = strcat(str1,num2str(N),' ,');

str2 = ('Iter = ');

str2 = strcat(str2,num2str(iter));

str = strcat(str1,str2);

title(str);
#!/usr/bin/perl -w
use strict;


my $ecut=0.001;
my @AA=qw(A  U  G  C -);
my %AA2index = ('A'=>'1', 'U'=>'2', 'G'=>'3', 'C'=>'4', '-'=>'5');

my $seq=$ARGV[0];
my $aln=$ARGV[1];
my $outfile=$ARGV[2];

my @seq=`cat $seq`; chomp(@seq);	
my $len=length $seq[1];

#print "parse ...\n";
my %freq=&wfreq($len, $aln);
	
my @nn=split(//, $seq[1]);
open(PRO, ">$outfile");
for(my $i=1; $i<=$len; $i++)
{
	print PRO "$nn[$i-1] ";
	foreach my $A(@AA)
	{
		printf PRO "%6d ", $freq{$i, $A};
	}
	    
	printf PRO "\n";
}
close(PRO);
	


sub wfreq
{
    my ($len, $file)=@_;

    my %ALN=();
    my $Pcount=0;
    open(ALN,"$file") || die "Cant open $file";
    while(my $line=<ALN>)
    {
        chomp($line);
        if($line =~ /^>(\S+)/)
        {
            my $Pname=$1;            
#            my $Evalue= $1 if($line =~ /E=(\S+)/);
#	    last if($Evalue>$ecut);
	    $Pcount++;
            $ALN{$Pcount, 0}=$Pname;
#            $ALN{$Pcount, 1}=$Evalue;
        }
        else
        {
	    $line =~ s/T/U/g;  ###replace T by U	    
	    if($line =~/(\S+)/){
            $ALN{$Pcount, 2}=$line;
	    }
        }
    }
    close(ALN);

    my %freq=();
	$Pcount=50000 if($Pcount>50000);
    printf "%d sequences\n", $Pcount;
    if($Pcount >= 1)
    {
        %freq = &frquency(\%ALN, $Pcount, \%AA2index);
    }
    else
    {
	my @Qres   = split(//, $ALN{1, 2});
	for(my $j=0; $j<@Qres; $j++)
	{
             foreach my $key (@AA)
             {
                 $freq{$j+1, $key}=0;
             }
	}
    }

    return %freq;
}


sub frquency
{
    my ($ALN_ref, $Nseq, $AA_ref)=@_;
    my %align   = %$ALN_ref;
    my %AA2in   = %$AA_ref;

    my @Qres   = split(//, $align{1, 2});
    my $Ncol   = $#Qres;
    my %res_count=();


    my $Qresno=0;
    my %Qmapping=();
    for(my $j=0; $j<=$#Qres; $j++)
    {
        $res_count{$j}=0;
        if($Qres[$j] ne '-')
        {
            $Qresno++;
            $Qmapping{$Qresno}=$j;
        }
    }


    my @ARR=();
    for(my $i=1; $i<=$Nseq; $i++)
    {
        my @res=split(//, $align{$i, 2});
        for(my $j=0; $j<=$#res; $j++)
        {
            $ARR[$i][$j]=$res[$j];
        }
    }
    my $AAcount = keys %AA2in;
    my %AA_freq=();
    my %sum_seq_weights=();
    my $k=0;

    for(my $j=0; $j<=$Ncol; $j++)
    {
        if($Qres[$j] eq '-')
        {
            next;
        }
        $k++;
        foreach my $key (@AA)
        {
            $AA_freq{$k, $key}=0;
        }
        my $w=0;
        for(my $i=1; $i<=$Nseq; $i++)
        {
            my $AAN="";
	    
            if(!exists $AA2in{$ARR[$i][$j]})
            {
		print "replace $ARR[$i][$j] by $ARR[1][$j]\n";
                $AAN=$ARR[1][$j]; #replace nonstandard base in templates by query base
            }
            else
            {
                $AAN=$ARR[$i][$j];
            }

#	    print "$AAN ";
            $AA_freq{$k, $AAN} += 1; ##weighted frequency in clolumn $j
        }
	#print "\n";
	
    }
    return %AA_freq;
}

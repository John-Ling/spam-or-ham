mod classifier;
use classifier::Classifier;
fn main() {
    let mut classifier = Classifier::new("1");
    let corpus = classifier.parse_csv("src/data/ling_cleaned.csv", "body", "label");
    let _ = classifier.train(corpus.unwrap());

    let _ = classifier.predict("
    
i am writing to ask alexis manaster ramer a simple question concerning the universality of certain phenomena : how come about a hundred languages are spoken in present-day russia despite the attempts to russify everyone ? this linguist seems to be very active , yet unfortunately he tends to generalize the information he is aware of , judging by his contribution to the discussion on he / she pronouns and his remarks concerning russian linguistic policies . i would like to tell him of a common example concerning various ethnicities in russia : in 1553 czar ivan the dreadful incorporated the mari , mordvin ( finnic ) , chuvash and tatars ( turkic ) nationalities into his state . nowadays these peoples live in the very heart of russia on the middle volga , and , apparently , enjoy their culture and languages . they have also contributed to the culture and language of the dominant nation . the infamous cossacks ( farmers living on the borders of the state and ready to resort to arms to protect them , a sort of michigan militia ) have preserved their language while acquiring a lot of culture from , say , mountaineers of the caucausus . yet they did not assimilate those peoples who still speak their languages . to draw parallels between nostratic universal processes and the language policies of russia might hardly be considered an appropriate example . it would rather be the opposite of the desired . valeri vassiliev st . john 's
    ");

    let _ = classifier.predict(
        "
        
the georgetown arabic department of the school of languages and linguistics will host intensive courses in modern standard arabic worth 12 undergraduate credits . elem , interm . , advanced and a course in arabic - eng . translation will be offered . for pamphlet , contact : belkacem baccouche , director , georgetown university , arabic dept . , washington , d . c . 20057-1082 . ( 202 ) 687-5743
        "
    );

    let _ = classifier.predict(
        "
   Hello folks
I'm new to Linux, so here goes...
to the outside through my .
    ",
    );

    let _ = classifier.predict("
   Hello all, Firstly I'd like to thank all of you for the fast and very helpful feedback that I got to my question today.  I have one more question though.  I downloaded the w3 and url files from the server at the first try thanks to the help that I received today.  Then though I tried to build them.  I started by trying the w3 program.  I used the following lines which produced some strange results.  Would any one be able to set me straight? ./configure --with-emacs --prefix=/usr/local/src/beta/w3 --exec-prefix=/usr/ local/src/beta/w3 --with-url=/url/url That worked fine so I moved to the next step. make At the bottem of the text I got the following messages: Cannot open load file: /url/url/url-vars.el make[1]: *** [custom-load.el] Error 255 make[1]: Leaving directory `/usr/local/src/beta/w3/lisp' make: *** [w3] Error 2 When I got around to trying the url package I had no problems.  In saying that this doesn't necessarily mean that I was doing it right so below are the commands I used. ./configure --with-emacs --prefix=/url/url --exec-prefix=/url/url followed by the commands make and make install. There is no text files which contain help on installing the url package so I'm not completely certain if I've used the right method here. Thanks again Darragh ----- Original Message ----- From: 
   'Hunt, Bryan'  To: ;  Cc:  Sent: Thursday, September 05, 2002 5:08 PM Subject: OT: RE: [ILUG] Newby to Linux looking for information on cvs > > speaking of that IDE (weblogic developer) the open source version called > eclipse is free from eclipse.org If you are doing java development you > need to get this IDE . I've been using it for the last month and it is > absolutly superb. Best thing about it is that rather than using swing > (which is crap) is that they have their own native widget set called swt. > > When you run it on windows it uses windows widgets but when you run it on > linux it uses gtk, you should see it on gnome 2 ! Absolutely stunning ! > > --B > > -----Original Message----- > From: Justin MacCarthy [mailto:macarthy@iol.ie] > Sent: 05 September 2002 16:53 > To: michael.conry@ucd.ie > Cc: ilug@linux.ie > Subject: RE: [ILUG] Newby to Linux looking for information on cvs > > > This is the best step by step guide to setting up cvs on Redhat.. > > http://www7b.software.ibm.com/wsdd/library/techarticles/0205_yu/yu.html?open > &l=456,t=gr > > It is for a particular IBM ide, but the setup and testing of the server is > the same for any CVS client > > Both The 'Using Linux' and 'Linux in a nutshell' book by Oreilly have > sections on cvs /rcs , and both books are a must buy for any linux newbie > > > Justin > > > > -----Original Message----- > > From: ilug-admin@linux.ie [mailto:ilug-admin@linux.ie]On Behalf Of > > Michael Conry > > Sent: 05 September 2002 16:34 > > To: Darragh > > Cc: ilug@linux.ie > > Subject: Re: [ILUG] Newby to Linux looking for information on cvs > > > > > > On 0020 +0100 %{!Thu, Sep 05, 2002 at  3:55:16PM +0100}, Darragh wrote: > > > Hello, > > > I am very new to Linux and need some help on a utility called > > cvs.  As far > > > as I'm aware its a similar protocol to FTP.  I need to use it > > to download a > > > program from :pserver:anoncvs@subversions.gnu.org:/cvsroot/w3. > > I am looking > > > for information on how to use it.  I'll have another look at > > the man pages > > > but I think I have to set it up before I can download anything. > > cvs is really a very different kind of thing to FTP, but the details of > > that statement are left as an exercise to the reader (won't show up my > > own ignorance that way ;-) > > The application you want is cvsclient... > > There is documentation here: > > http://www.fokus.gmd.de/gnu/docs/cvs/cvsclient_toc.html > > > > You might get a quick idea of how it works from here: > > http://www.sci.muni.cz/~mikulik/gnuplot.html > > where he explains how to get cvs gnuplot... > > The commands are: > > > > export > > CVSROOT=:pserver:anonymous@cvs.gnuplot.sourceforge.net:/cvsroot/gnuplot > > cvs login > > cvs -z3 checkout gnuplot > > > > Something similar will probably do the job for you.  I'm guessing the > > following MIGHT work... > > > > export CVSROOT=:pserver:anoncvs@subversions.gnu.org:/cvsroot/w3 > > cvs login > > cvs -z3 checkout w3 > > > > m > > -- > > Michael Conry   Ph.:+353-1-7161987, Web: http://www.acronymchile.com > > Key fingerprint = 5508 B563 6791 5C84 A947  CB01 997B 3598 09DE 502C > > > > -- > > Irish Linux Users' Group: ilug@linux.ie > > http://www.linux.ie/mailman/listinfo/ilug for (un)subscription > > information. > > List maintainer: listmaster@linux.ie > > > > > > > > > -- > Irish Linux Users' Group: ilug@linux.ie > http://www.linux.ie/mailman/listinfo/ilug for (un)subscription information. > List maintainer: listmaster@linux.ie > > -- > Irish Linux Users' Group: ilug@linux.ie > http://www.linux.ie/mailman/listinfo/ilug for (un)subscription information. > List maintainer: listmaster@linux.ie > --  Irish Linux Users' Group: ilug@linux.ie http://www.linux.ie/mailman/listinfo/ilug for (un)subscription information. List maintainer: listmaster@linux.ie kj"
    );

    println!("TESTING SPAM");
    let _ = classifier.predict("
        adults only click this http : / / www . wildhotsex . com / puresex1 . html nothing  butt  pure sex this adult site gives you a free xxx video every month http : / / www . wildhotsex . com / puresex1 . html to be removed - e-mail jillbob622 @ geocities . com put remove in the subject line .
    ");

    let _ = classifier.predict(
        "
    WE NEED HELP.  We are a 14 year old fortune 500 company, and we have 
grown 1000%!  We cannot keep up.  We are looking for individuals who 
want to work at home, and make a good living.

So if you are looking to be employed from home with a career that has
vast opportunities, then go:

http://www.lotsonet.com/opportunity

and fill out our info form.  NO EXPERIENCE REQUIRED, WE WILL TRAIN YOU.
NO COMMITTMENT IS REQUIRED BY FILLING OUT THE FORM, IT IS FOR INFO
ONLY.

http://www.lotsonet.com/opportunity

You want to be independent?  THEN MAKE IT HAPPEN!
HAPPEN!

SIMPLY CLICK ON THE LINK BELOW FOR FREE, NO OBLIGATED INFORMATION!
GUARANTEED!

http://www.lotsonet.com/opportunity


To be removed from our link simple go to:

http://www.lotsonet.com/opportunity/remove.html


8064owaS8-626pYyF6611Pnah5-739Lind4606qvym4-041dhmY3817l52
    ",
    );

    let _ = classifier.predict("
        A man endowed with a 7-8'' hammer is simply  better equipped than a man with a 5-6'' hammer.  Would you rather havemore than enough to get the job done or fall = short. It's totally upto you. Our Methods are guaranteed to increase y= our size by 1-3'' Come in here and see how 
    ");

    let _ = classifier.predict("
        make $ 3500 per week using your home computer ! put my free software in your computer . . . . . start making huge amounts of cash . . . . . without working ! ! ! ! your computer does all the work . . you get all the money ! ! ! you never have to talk to anybody ! ! just load my free software into your computer . . . and let it go . . . you will not believe how much money will come rolling in because of the efforts of your computer ! ! ! to get more information on this email me at . . manboca @ hotmail . com

     ");
}

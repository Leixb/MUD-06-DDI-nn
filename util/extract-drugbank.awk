#! /usr/bin/awk -f

/<drug / { indrug = 1; }

/<products>/ { indrug = 0; inprod=1; }
/<.products>/ { inprod=0; }
/<categories>/ { incat=1; }
/<.categories>/ { incat=0; }
 
/<name>/ {  i=index($0,"<name>")+6
            f=index($0,"</name>")-1
            name = substr($0,i,f-i+1)
            type = ""
            if (indrug) type="|drug"
            else if (inprod) type="|brand"
            if (type!="") print name type
         }

/<category>.*<.category>/ {
            if (incat) {
               i=index($0,"<category>")+10
               f=index($0,"</category>")-1
               name = substr($0,i,f-i+1)
               type = "|group"
               print name type
	    }
         }

